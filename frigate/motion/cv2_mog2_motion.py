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
        self._lut = np.zeros(256, np.uint8)  # TODO delete me
        self._prev_boxes: list[tuple[tuple[int, int, int, int], int]] = []  # TODO delete me
        self._frame_idx = 0
        self._warned_low_height: int | None = None  # TODO delete me
        self._ocl = getattr(cv2, "ocl", None)
        self._use_ocl = self._probe_opencl()
        # cached device-side UMat objects for the fused GPU pipeline
        self._inv_mask_umat: cv2.UMat | None = None
        self._kernel_umat: cv2.UMat | None = None
        self._kernel_size_cached: int | None = None
        self.calibrating = True
        self.update_mask()

    def is_calibrating(self) -> bool:
        return self.calibrating

    def _build_input(self, frame: np.ndarray) -> np.ndarray:
        """Build the full-resolution MOG2 input from the I420 frame.

        Frigate hands detect() a YUV420p (I420) buffer: a single 2-D uint8
        array of shape (H*3//2, W) with the Y plane stacked over the U/V
        planes. The Y-plane slice already is grayscale (no cvtColor needed);
        when use_bgr is set the whole I420 frame is converted to BGR so MOG2
        models 3-channel color instead of luma.
        """
        if self._use_bgr:
            return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
        H, W = self.frame_shape
        return frame[0:H, 0:W]

    def detect_new(self, frame: np.ndarray): -> list[tuple[int, int, int, int]]:
        # WIP cleanup version
        # fast return if ptz_moving

        # umat version just sends frame to umat + uses umat inv_mask, kernel
        # downsample
        # mask
        # run model
        # morphology (TODO do we keep this?)
        # get contours
        # handle skip_motion_threshold / lightning threshold



    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        # with an OpenCL platform the fused GPU pipeline handles the frame;
        # once a runtime fault has permanently disabled the GPU, this
        # method's own CPU body below serves the rest of the detector's life
        if self._use_ocl:
            return self.detect_ocl(frame)

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

        small = cv2.resize(
            self._build_input(frame),
            dsize=(self._proc_size[1], self._proc_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        # optional percentile contrast norm
        # this has to come before masking so excluded pixels (0) cannot
        # drag the min percentile toward 0 (matches the stock ordering).
        # Skipped while use_bgr is set: the percentile window and LUT are
        # tuned for the single-channel luma plane, so they are not applied
        # to 3-channel color input for now
        if self._contrast_enabled and not self._use_bgr:
            small = self._normalize_contrast(small)

        small = cv2.bitwise_and(small, small, mask=self._inv_mask)

        # feed the model and grab the foreground plane (0=bg, ~127=shadow,
        # 255=fg)
        fg_model = self._sub.apply(small, learningRate=self._effective_rate())
        self._frame_idx += 1
        if self.calibrating and self._frame_idx < self._warmup_frames:
            # warmup: keep learning the background, emit nothing
            return motion_boxes

        # shadow handling: value-robust (OpenCV shadow may be 127,
        # historically 125)
        if self._shadow_mode == "keep":
            fg = cv2.threshold(fg_model, 0, 255, cv2.THRESH_BINARY)[1]
        else:
            fg = cv2.inRange(fg_model, 255, 255)  # type: ignore[call-overload]

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

    def detect_ocl(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Fused GPU pipeline variant of detect() for OpenCL platforms.

        Uploads the input once (the luma plane, or the raw I420 buffer when
        use_bgr is set, in which case the I420 -> BGR conversion also runs
        on-device); resize, contrast normalization, ROI masking, MOG2
        apply, shadow thresholding and morphology all run on the GPU
        (chained UMat keeps intermediates on-device). Only the final
        foreground mask is copied back, since findContours has no OpenCL
        implementation. Per-frame CPU work is reduced to contour
        extraction, the persistence gate and the percentile math. On any
        cv2.error the detector permanently falls back to the CPU path
        (see _disable_ocl for the contract).
        """
        motion_boxes: list[tuple[int, int, int, int]] = []

        if not self.config.enabled:
            return motion_boxes

        # GPU disabled at init or after a runtime fault: same contract,
        # pure CPU pipeline (UMat round trips would buy nothing)
        if not self._use_ocl:
            return self.detect(frame)

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

        try:
            # single H2D upload; everything from here until the final
            # .get() stays on the GPU (the UMat ctor stubs only cover
            # UMat args; ndarray is accepted at runtime)
            if self._use_bgr:
                # use_bgr: upload the raw I420 buffer and run the
                # I420 -> BGR conversion on-device (a UMat input is what
                # routes cvtColor through its OpenCL path), so the
                # 3-channel frame never round-trips to the host. The
                # OCL flag is always set before any UMat exists (init
                # probe), which the driver requires for the on-device
                # buffer handoff
                small: cv2.UMat = cv2.UMat(frame)  # type: ignore[call-overload]
                small = cv2.cvtColor(small, cv2.COLOR_YUV2BGR_I420)
            else:
                H, W = self.frame_shape
                small = cv2.UMat(frame[0:H, 0:W])  # type: ignore[call-overload]
            small = cv2.resize(
                small,
                dsize=(self._proc_size[1], self._proc_size[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            # optional percentile contrast norm
            # (before masking, same ordering as the CPU path, which skips
            # it while use_bgr is set)
            if self._contrast_enabled and not self._use_bgr:
                small = self._normalize_contrast_ocl(small)

            small = cv2.bitwise_and(small, small, mask=self._get_inv_mask_umat())

            # feed the model and grab the foreground plane (0=bg,
            # ~127=shadow, 255=fg); the UMat entry point is the GPU
            fg_model = self._sub.apply(small, learningRate=self._effective_rate())
            self._frame_idx += 1
            if self.calibrating and self._frame_idx < self._warmup_frames:
                # warmup: keep learning the background, emit nothing
                return motion_boxes

            # shadow handling: value-robust (OpenCV shadow may be 127,
            # historically 125)
            if self._shadow_mode == "keep":
                fg = cv2.threshold(fg_model, 0, 255, cv2.THRESH_BINARY)[1]
            else:
                fg = cv2.inRange(fg_model, 255, 255)  # type: ignore[call-overload]

            # optional morphology open/close (scrub speckle)
            if self._morphology.enabled:
                kernel = self._get_kernel_umat()
                iterations = self._morphology.iterations
                fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=iterations)
                fg = cv2.morphologyEx(
                    fg, cv2.MORPH_CLOSE, kernel, iterations=iterations
                )

            # zero excluded regions on-device (matches the numpy
            # scatter fg[self._mask] = 0 in the CPU path)
            fg = cv2.bitwise_and(fg, fg, mask=self._get_inv_mask_umat())

            # single D2H: only the final fg mask returns to the CPU
            fg_host = fg.get()
        except cv2.error as err:
            # runtime GPU fault: switch to the CPU path for the rest of
            # this detector's life (a failed apply may leave the model
            # inconsistent, so _disable_ocl rebuilds it) and serve this
            # frame from the CPU pipeline
            self._disable_ocl(err)
            return self.detect(frame)

        # contours -> boxes in proc space (area gates; min defaults to
        # the contour_area setting, which is on the same pixel scale)
        min_area = self._contours.min_area or self.config.contour_area or 0
        max_area = (
            self._proc_size[0] * self._proc_size[1] * self._contours.max_area_ratio
        )
        proc_boxes: list[tuple[int, int, int, int]] = []
        contours = grab_cv2_contours(
            cv2.findContours(fg_host, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        # is forced
        pct_motion = cv2.countNonZero(fg_host) / (
            self._proc_size[0] * self._proc_size[1]
        )
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
        self._use_bgr = m.use_bgr
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
        # invalidate cached device-side masks/kernels for the fused pipeline
        self._inv_mask_umat = None
        self._kernel_umat = None
        self._kernel_size_cached = None

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

    def _effective_rate(self) -> float:
        """The learning rate MOG2 apply receives for the next frame.

        Unset rates (the default) resolve to the negative value that
        documents MOG2's automatic adaptive rate; a fixed low rate absorbs
        stationary objects into the background quickly and measurably
        reduces motion recall (74% -> 15% trigger on the eval clip).
        """
        rate = self._calibration_rate if self.calibrating else self._learning_rate
        return -1.0 if rate is None else rate

    def _get_inv_mask_umat(self) -> cv2.UMat:
        """Cached device-side copy of the ROI mask (rebuilt by update_mask)."""
        if self._inv_mask_umat is None:
            self._inv_mask_umat = cv2.UMat(self._inv_mask)
        return self._inv_mask_umat

    def _get_kernel_umat(self) -> cv2.UMat:
        """Cached device-side morphology kernel (rebuilt on size change)."""
        kernel = self._kernel_umat
        if kernel is None or self._kernel_size_cached != self._morphology.kernel_size:
            size = self._morphology.kernel_size
            self._kernel_size_cached = size
            new_kernel: cv2.UMat = cv2.UMat(
                cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
            )  # type: ignore[call-overload]
            self._kernel_umat = new_kernel
            kernel = new_kernel
        return kernel

    def _disable_ocl(self, err: Exception) -> None:
        """Fall back to the CPU path for the lifetime of this detector."""
        logger.warning(
            "%s: OpenCL MOG2 apply failed (%s); falling back to CPU",
            self.name,
            err,
        )
        self._use_ocl = False
        if self._ocl is not None:
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
        if self._contrast_rescale(hist, frame.size):
            frame = cv2.LUT(frame, self._lut)
        return frame

    def _contrast_rescale(self, hist: np.ndarray, total: int) -> bool:
        """Update the percentile window and LUT from a 256-bin histogram.

        Shared by the CPU and fused GPU contrast paths so both keep the
        same moving-window state. Returns True when a LUT rescale was
        applied (self._lut holds the new LUT).
        """
        cum_hist = np.cumsum(hist)
        min_value = np.searchsorted(
            cum_hist, total * (self._contrast_min_pct / 100.0)
        ).astype(np.uint8)
        max_value = np.searchsorted(
            cum_hist, total * (self._contrast_max_pct / 100.0)
        ).astype(np.uint8)
        # skip contrast calcs if the image is a single color
        if min_value >= max_value:
            return False
        # keep track of the last N contrast values
        self._contrast_values[self._contrast_index] = [min_value, max_value]
        self._contrast_index += 1
        if self._contrast_index == len(self._contrast_values):
            self._contrast_index = 0

        avg_min, avg_max = np.mean(self._contrast_values, axis=0)

        # LUT rescale replaces np.clip + per-pixel math
        bins = np.arange(256)
        lut = np.clip((bins - avg_min) * (255.0 / (avg_max - avg_min + 1e-6)), 0, 255)
        self._lut = lut.astype(np.uint8)
        return True

    def _normalize_contrast_ocl(self, frame: cv2.UMat) -> cv2.UMat:
        """GPU twin of _normalize_contrast: histogram and LUT on the device.

        Only the 256-bin histogram comes back to the host for the
        percentile math (shared with the CPU path via _contrast_rescale).
        The cv2.UMat wrapper exposes no pixel count, so the total comes
        from the known processing size.
        """
        hist = cv2.calcHist([frame], [0], None, [256], [0, 256]).get().flatten()
        if self._contrast_rescale(hist, self._proc_size[0] * self._proc_size[1]):
            # the LUT stubs require a UMat lut; the ndarray form is accepted
            # at runtime (verified on-device) and avoids a per-frame upload
            frame = cv2.LUT(frame, self._lut)  # type: ignore[call-overload]
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

        matched: list[int | None] = [None] * len(boxes)
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
            m = matched[i]
            streak = prev_streaks[m] + 1 if m is not None else 1
            new_tracked.append((box, streak))
            if streak >= self._persistence_frames:
                out.append(box)
        self._prev_boxes = new_tracked
        return out
