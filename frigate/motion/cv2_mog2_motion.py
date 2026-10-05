"""OpenCV MOG2 background subtraction motion detector."""

import logging

import cv2
import numpy as np

from frigate.camera import PTZMetrics
from frigate.config.config import RuntimeMotionConfig
from frigate.motion import MotionDetector
from frigate.util.image import grab_cv2_contours

logger = logging.getLogger(__name__)


class Cv2Mog2MotionDetector(MotionDetector):
    """Motion detector based on cv2 MOG2 background subtraction.

    Operates on the luma plane downscaled to frame_height, applies an
    optional percentile contrast normalization, and emits boxes in
    full-frame (x1, y1, x2, y2) pixels, the same contract as
    ImprovedMotionDetector.

    A temporal persistence gate only emits boxes for regions that have
    been present for persistence_frames consecutive frames, tracked
    across frames by center distance so a moving object keeps its
    streak while a flickering patch does not.
    """

    # tuned MOG2 processing height; the stock frame_height default (100)
    # is too coarse for MOG2 (motion recall and dappled suppression fail)
    _DEFAULT_FRAME_HEIGHT = 360
    # below this, MOG2 loses the detail it needs; log a warning
    _MIN_RECOMMENDED_FRAME_HEIGHT = 200

    def __init__(
        self,
        frame_shape: tuple[int, int],
        config: RuntimeMotionConfig,
        fps: int,
        name: str = "mog2",
        ptz_metrics: PTZMetrics | None = None,
    ) -> None:
        self.name = name
        self.config = config
        self.frame_shape = frame_shape
        self.ptz_metrics = ptz_metrics
        self._lut = np.zeros(256, np.uint8)
        self._prev_boxes: list[tuple[tuple[int, int, int, int], int]] = []
        self._frame_idx = 0
        self._warned_low_height: int | None = None
        self._ocl = getattr(cv2, "ocl", None)
        self._use_ocl = self._probe_opencl()
        self.calibrating = True
        self.update_mask()

    def is_calibrating(self) -> bool:
        return self.calibrating

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        motion_boxes: list[tuple[int, int, int, int]] = []

        if not self.config.enabled:
            return motion_boxes

        # if ptz motor is moving from autotracking, quickly return
        # a single box that is 80% of the frame
        if self._ptz_moving():
            return [
                (
                    int(self.frame_shape[1] * 0.1),
                    int(self.frame_shape[0] * 0.1),
                    int(self.frame_shape[1] * 0.9),
                    int(self.frame_shape[0] * 0.9),
                )
            ]

        H, W = self.frame_shape
        gray = frame[0:H, 0:W]

        small = cv2.resize(
            gray,
            dsize=(self._proc_size[1], self._proc_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        # optional percentile contrast norm
        # this has to come before masking so excluded pixels (0) cannot
        # drag the min percentile toward 0 (matches the stock ordering)
        if self._contrast_enabled:
            small = self._normalize_contrast(small)

        small = cv2.bitwise_and(small, small, mask=self._inv_mask)

        # feed the model and grab the foreground plane (0=bg, ~127=shadow,
        # 255=fg); while calibrating, adapt the background faster, mirroring
        # the stock detector's 0.2 calibration blend
        learning_rate = (
            self._calibration_rate if self.calibrating else self._learning_rate
        )
        if self._use_ocl:
            try:
                # iGPU path: UMat in/out around the only GPU-bound step;
                # resize/contrast/mask/morphology/contours stay on the CPU
                fg_model = self._apply_ocl(small, learning_rate)
            except cv2.error as err:
                # runtime GPU fault: switch to CPU for the rest of this
                # detector's life (a failed apply may leave the model
                # inconsistent, so _disable_ocl rebuilds it)
                self._disable_ocl(err)
                fg_model = self._sub.apply(small, learning_rate)
        else:
            fg_model = self._sub.apply(small, learning_rate)
        self._frame_idx += 1
        if self.calibrating and self._frame_idx < self._warmup_frames:
            # warmup: keep learning the background, emit nothing
            return motion_boxes

        # shadow handling: value-robust (OpenCV shadow may be 127,
        # historically 125)
        if self._shadow_mode == "keep":
            fg = cv2.threshold(fg_model, 0, 255, cv2.THRESH_BINARY)[1]
        else:
            fg = cv2.inRange(fg_model, 255, 255)

        # optional morphology open/close (scrub speckle)
        if self._morphology.enabled:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (self._morphology.kernel_size, self._morphology.kernel_size),
            )
            fg = cv2.morphologyEx(
                fg,
                cv2.MORPH_OPEN,
                kernel,
                iterations=self._morphology.iterations,
            )
            fg = cv2.morphologyEx(
                fg,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=self._morphology.iterations,
            )

        fg[self._mask] = 0

        # contours -> boxes in proc space (area gates; min defaults to the
        # contour_area setting, which is on the same pixel scale)
        min_area = self._contours.min_area or self.config.contour_area or 0
        max_area = (
            self._proc_size[0] * self._proc_size[1] * self._contours.max_area_ratio
        )
        proc_boxes: list[tuple[int, int, int, int]] = []
        contours = grab_cv2_contours(
            cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        )
        for c in contours:
            contour_area = cv2.contourArea(c)
            if min_area <= contour_area <= max_area:
                x, y, w, h = cv2.boundingRect(c)
                proc_boxes.append((x, y, w, h))

        # persistence gate (proc space, tracked by center distance)
        proc_boxes = self._filter_persistent(proc_boxes)

        # scale to full frame -> (x1, y1, x2, y2)
        rf = self._resize_factor
        motion_boxes = [
            (
                int(x * rf),
                int(y * rf),
                int((x + w) * rf),
                int((y + h) * rf),
            )
            for (x, y, w, h) in proc_boxes
        ]

        # skip motion entirely if the scene change percentage exceeds the
        # configured threshold; the frame is dropped and a recalibration
        # is forced. note: pct is the post-morphology, post-mask fg-pixel
        # fraction, which the stock detector computes from contour area
        pct_motion = cv2.countNonZero(fg) / (self._proc_size[0] * self._proc_size[1])
        if (
            self.config.skip_motion_threshold is not None
            and pct_motion > self.config.skip_motion_threshold
        ):
            self.calibrating = True
            self._reset_state()
            return []

        # once the motion is less than 5% and the number of boxes is < 4,
        # assume it is calibrated
        if pct_motion < 0.05 and len(motion_boxes) <= 4:
            self.calibrating = False

        # on a large scene change (lightning, ir, ptz) relearn the
        # background at the calibration learning rate; no model rebuild
        if self.calibrating or pct_motion > self.config.lightning_threshold:
            self.calibrating = True
            if pct_motion > self.config.lightning_threshold:
                logger.debug(
                    "%s: large scene change, recalibrating MOG2 background",
                    self.name,
                )
                self._reset_state()

        return motion_boxes

    def update_mask(self) -> None:
        """Update the motion mask and relearn the background after a config change."""
        m = self.config.mog2

        # re-read all MOG2 knobs so hot-reload of any setting is honored
        self._history = m.history
        self._var_threshold = m.var_threshold
        self._learning_rate = m.learning_rate
        self._calibration_rate = m.calibration_learning_rate
        self._shadow_mode = m.shadow_mode
        self._contrast_enabled = m.contrast_norm
        self._contrast_history = m.contrast_history
        self._contrast_min_pct = m.contrast_min_pct
        self._contrast_max_pct = m.contrast_max_pct
        self._persistence_frames = m.persistence_frames
        self._persistence_tolerance = m.persistence_match_tolerance
        self._morphology = m.morphology
        self._contours = m.contours
        self._warmup_frames = m.warmup_frames

        # downscale to the MOG2 processing height. the stock frame_height
        # default (100) is too coarse for MOG2, so the tuned default (360)
        # is used unless the user set mog2.frame_height explicitly
        H, W = self.frame_shape
        if m.frame_height is not None:
            frame_height = m.frame_height
            if (
                frame_height < self._MIN_RECOMMENDED_FRAME_HEIGHT
                and self._warned_low_height != frame_height
            ):
                logger.warning(
                    "%s: mog2.frame_height %d is below the recommended %d; "
                    "motion sensitivity and dappled suppression degrade at "
                    "low processing resolutions",
                    self.name,
                    frame_height,
                    self._DEFAULT_FRAME_HEIGHT,
                )
                self._warned_low_height = frame_height
        else:
            frame_height = min(self._DEFAULT_FRAME_HEIGHT, H)
        self._resize_factor = H / frame_height
        self._proc_size = (frame_height, round(frame_height * W / H))

        # reset the contrast state (moving min/max percentile window;
        # column 1 (max) starts at 255 so early frames behave sanely)
        self._contrast_values = np.zeros((m.contrast_history, 2), np.uint8)
        self._contrast_values[:, 1:2] = 255
        self._contrast_index = 0

        resized_mask = cv2.resize(
            self.config.rasterized_mask,
            dsize=(self._proc_size[1], self._proc_size[0]),
            interpolation=cv2.INTER_AREA,
        )
        excluded = resized_mask == 0
        self._mask = np.where(excluded)
        self._inv_mask = (~excluded).astype(np.uint8) * 255

        # reset detection state and relearn the background with the new
        # mask and parameters
        self._build_model()
        self.calibrating = True
        self._frame_idx = 0
        self._prev_boxes = []

    def stop(self) -> None:
        """Stop the motion detector."""
        pass

    def _build_model(self) -> None:
        # detectShadows=True always: the shadow plane is needed so
        # shadow_mode can split it
        self._sub = cv2.createBackgroundSubtractorMOG2(
            history=self._history,
            varThreshold=self._var_threshold,
            detectShadows=True,
        )

    def _probe_opencl(self) -> bool:
        """Greedily probe for an OpenCL platform (e.g. an Intel iGPU).

        cv2's useOpenCL flag is process-global, but the detect process is
        per-camera, so enabling it here only affects this camera's pipeline;
        operations without an OpenCL kernel fall back to CPU transparently.
        """
        if self._ocl is None or not self._ocl.haveOpenCL():
            return False
        try:
            self._ocl.setUseOpenCL(True)
            logger.info(
                "%s: OpenCL platform available, accelerating MOG2 apply on the GPU",
                self.name,
            )
            return True
        except cv2.error as err:
            logger.warning(
                "%s: OpenCL enable failed (%s); using CPU for MOG2", self.name, err
            )
            return False

    def _apply_ocl(self, frame: np.ndarray, learning_rate: float) -> np.ndarray:
        """Run MOG2 apply on the GPU and copy the foreground plane back."""
        return self._sub.apply(cv2.UMat(frame), learning_rate).get()

    def _disable_ocl(self, err: Exception) -> None:
        """Fall back to the CPU path for the lifetime of this detector."""
        logger.warning(
            "%s: OpenCL MOG2 apply failed (%s); falling back to CPU",
            self.name,
            err,
        )
        self._use_ocl = False
        self._ocl.setUseOpenCL(False)
        # restart warmup and persistence on a fresh model
        self._build_model()
        self._frame_idx = 0
        self._prev_boxes = []

    def _reset_state(self) -> None:
        # restart the warmup and persistence state without rebuilding the
        # model; the calibration learning rate relearns the scene
        self._prev_boxes = []
        self._frame_idx = 0

    def _ptz_moving(self) -> bool:
        return (
            self.ptz_metrics is not None
            and self.ptz_metrics.autotracker_enabled.value
            and not self.ptz_metrics.motor_stopped.is_set()
        )

    def _normalize_contrast(self, frame: np.ndarray) -> np.ndarray:
        """Apply percentile contrast normalization via a cv2 LUT rescale."""
        # histogram-accelerated percentile (replaces np.percentile)
        hist = cv2.calcHist([frame], [0], None, [256], [0, 256]).flatten()
        cum_hist = np.cumsum(hist)
        total = frame.size
        min_value = np.searchsorted(
            cum_hist, total * (self._contrast_min_pct / 100.0)
        ).astype(np.uint8)
        max_value = np.searchsorted(
            cum_hist, total * (self._contrast_max_pct / 100.0)
        ).astype(np.uint8)
        # skip contrast calcs if the image is a single color
        if min_value < max_value:
            # keep track of the last N contrast values
            self._contrast_values[self._contrast_index] = [min_value, max_value]
            self._contrast_index += 1
            if self._contrast_index == len(self._contrast_values):
                self._contrast_index = 0

            avg_min, avg_max = np.mean(self._contrast_values, axis=0)

            # LUT rescale replaces np.clip + per-pixel math
            bins = np.arange(256)
            lut = np.clip(
                (bins - avg_min) * (255.0 / (avg_max - avg_min + 1e-6)), 0, 255
            )
            self._lut = lut.astype(np.uint8)
            frame = cv2.LUT(frame, self._lut)

        return frame

    def _filter_persistent(
        self, boxes: list[tuple[int, int, int, int]]
    ) -> list[tuple[int, int, int, int]]:
        """Keep only boxes present for at least persistence_frames.

        Boxes are tracked across frames by center distance so a moving
        object keeps its streak while a flickering patch does not.
        """
        if self._persistence_frames <= 0:
            return boxes

        prev_boxes = [b for (b, _streak) in self._prev_boxes]
        prev_streaks = [s for (_box, s) in self._prev_boxes]

        # greedy nearest-center matching, most confident (closest) pairs first
        pairs: list[tuple[float, int, int]] = []
        for i, box in enumerate(boxes):
            cx1 = box[0] + box[2] / 2.0
            cy1 = box[1] + box[3] / 2.0
            for j, prev in enumerate(prev_boxes):
                cx2 = prev[0] + prev[2] / 2.0
                cy2 = prev[1] + prev[3] / 2.0
                pairs.append(((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2, i, j))
        pairs.sort(key=lambda p: p[0])

        matched = [None] * len(boxes)
        used = [False] * len(prev_boxes)
        for dist2, i, j in pairs:
            if matched[i] is not None or used[j]:
                continue
            box = boxes[i]
            prev = prev_boxes[j]
            tol = self._persistence_tolerance * (
                max(box[2], box[3]) + max(prev[2], prev[3])
            )
            if dist2 <= tol * tol:
                matched[i] = j
                used[j] = True

        out: list[tuple[int, int, int, int]] = []
        new_tracked: list[tuple[tuple[int, int, int, int], int]] = []
        for i, box in enumerate(boxes):
            streak = prev_streaks[matched[i]] + 1 if matched[i] is not None else 1
            new_tracked.append((box, streak))
            if streak >= self._persistence_frames:
                out.append(box)
        self._prev_boxes = new_tracked
        return out
