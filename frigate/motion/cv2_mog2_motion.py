"""OpenCV MOG2 background subtraction motion detector."""

import logging

import cv2
import numpy as np

from frigate.camera import PTZMetrics
from frigate.config.config import RuntimeMotionConfig
from frigate.motion import MotionDetector
from frigate.util.image import grab_cv2_contours

logger = logging.getLogger(__name__)

# a frame is a host array, or a UMat when OpenCL runs
Frame = cv2.UMat | np.ndarray


class Cv2Mog2MotionDetector(MotionDetector):
    """MOG2 background subtraction motion detector.

    Boxes are full-frame (x1, y1, x2, y2). The same steps run on host arrays or UMat.
    """

    # MOG2 needs more detail than the default frame height gives
    _DEFAULT_FRAME_HEIGHT = 360
    # below this, MOG2 misses detail
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
        self._use_ocl = self._probe_opencl()
        # cached ROI mask and morphology kernel (device copies when OpenCL runs)
        self._inv_mask_umat: cv2.UMat | None = None
        self._kernel: Frame | None = None
        self._kernel_size_cached: int | None = None
        # fallback box while the PTZ motor moves: 80% of the frame
        H, W = self.frame_shape
        self._ptz_box: tuple[int, int, int, int] = (
            int(W * 0.1),
            int(H * 0.1),
            int(W * 0.9),
            int(H * 0.9),
        )
        self.calibrating = True
        self.update_mask()

    def is_calibrating(self) -> bool:
        return self.calibrating

    def _build_input(self, frame: np.ndarray) -> np.ndarray:
        """The MOG2 input at full resolution: luma, or BGR when use_bgr is set."""
        if self._use_bgr:
            return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
        return self._luma_plane(frame)

    def _luma_plane(self, frame: np.ndarray) -> np.ndarray:
        """The Y plane of the I420 buffer, already grayscale."""
        H, W = self.frame_shape
        return frame[0:H, 0:W]

    def _to_device(self, frame: np.ndarray) -> Frame:
        """Return the frame in the pipeline's representation: host array or UMat."""
        if not self._use_ocl:
            return self._build_input(frame)
        if self._use_bgr:
            # a UMat input keeps the color conversion on the GPU
            i420: Frame = cv2.UMat(frame)  # type: ignore[call-overload]
            return cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420)
        return cv2.UMat(self._luma_plane(frame))  # type: ignore[call-overload]

    def _to_host(self, image: Frame) -> np.ndarray:
        """Copy a UMat back to the host; host arrays pass through."""
        if isinstance(image, cv2.UMat):
            return image.get()
        return image

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect motion and return boxes in full-frame (x1, y1, x2, y2) pixels.

        The same steps run on host arrays or on UMat; only the borders differ.
        """
        if not self.config.enabled:
            return []

        # the PTZ motor is moving, so the whole view changed
        if self._ptz_moving():
            return self._ptz_box

        small = self._to_device(frame)
        small = cv2.resize(
            small,
            dsize=(self._proc_size[1], self._proc_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        # luma only, and before masking so masked pixels do not skew the percentiles
        if self._contrast_enabled and not self._use_bgr:
            small = self._normalize_contrast(small)

        # zero excluded regions before the model sees the frame
        small = self._exclude_roi(small)

        # MOG2 output: 0 background, ~127 shadow, 255 foreground
        fg = self._sub.apply(small, learningRate=self._effective_rate())
        self._frame_idx += 1
        if self.calibrating and self._frame_idx < self._warmup_frames:
            # warmup feeds the model but emits no boxes
            return []

        # keep mode takes any non-zero value; shadow values vary by build
        if self._shadow_mode == "keep":
            fg = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY)[1]
        else:
            fg = cv2.inRange(fg, 255, 255)  # type: ignore[call-overload]

        # open/close removes speckle
        if self._morphology.enabled:
            kernel = self._morph_kernel()
            iterations = self._morphology.iterations
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=iterations)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=iterations)

        # drop excluded regions from the foreground
        fg = self._exclude_roi(fg)

        # one device-to-host copy per frame
        return self._evaluate_foreground(self._to_host(fg))

    def _evaluate_foreground(self, fg: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Turn a host foreground mask into boxes and update calibration state."""
        # area gates are in processing pixels
        min_area = self._contours.min_area or self.config.contour_area or 0
        max_area = (
            self._proc_size[0] * self._proc_size[1] * self._contours.max_area_ratio
        )
        proc_boxes: list[tuple[int, int, int, int]] = []
        for c in grab_cv2_contours(
            cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ):
            contour_area = cv2.contourArea(c)
            if min_area <= contour_area <= max_area:
                x, y, w, h = cv2.boundingRect(c)
                proc_boxes.append((x, y, w, h))

        proc_boxes = self._filter_persistent(proc_boxes)

        # processing (x, y, w, h) to full-frame (x1, y1, x2, y2)
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

        # foreground fraction after masking and morphology
        pct_motion = cv2.countNonZero(fg) / self._proc_pixels

        # too much change: drop the frame and recalibrate
        if (
            self.config.skip_motion_threshold is not None
            and pct_motion > self.config.skip_motion_threshold
        ):
            self.calibrating = True
            self._reset_state()
            return []

        # a quiet scene ends calibration
        if pct_motion < 0.05 and len(motion_boxes) <= 4:
            self.calibrating = False

        # a large scene change restarts learning
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

        # read every setting so config changes apply here
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

        # MOG2 needs a higher processing height than the config default
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
        self._proc_pixels = self._proc_size[0] * self._proc_size[1]

        # moving min/max window; max starts at 255 so early frames pass
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
        # the cached copies are now stale
        self._inv_mask_umat = None
        self._kernel = None
        self._kernel_size_cached = None

        # restart background learning with the new mask
        self._build_model()
        self.calibrating = True
        self._frame_idx = 0
        self._prev_boxes = []

    def stop(self) -> None:
        """Stop the motion detector."""
        pass

    def _build_model(self) -> None:
        # shadows stay on so shadow_mode can split them out
        self._sub = cv2.createBackgroundSubtractorMOG2(
            history=self._history,
            varThreshold=self._var_threshold,
            detectShadows=True,
        )

    def _probe_opencl(self) -> bool:
        """Enable OpenCL when a platform exists. The cv2 flag is global."""
        ocl = getattr(cv2, "ocl", None)
        if ocl is None or not ocl.haveOpenCL():
            return False
        try:
            ocl.setUseOpenCL(True)
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
        """The MOG2 learning rate for the next frame; None means MOG2's
        adaptive rate, sent as -1.0.
        """
        rate = self._calibration_rate if self.calibrating else self._learning_rate
        return -1.0 if rate is None else rate

    def _roi_mask(self) -> Frame:
        """The ROI mask in the pipeline's representation (device copy cached)."""
        if not self._use_ocl:
            return self._inv_mask
        if self._inv_mask_umat is None:
            self._inv_mask_umat = cv2.UMat(self._inv_mask)
        return self._inv_mask_umat

    def _exclude_roi(self, image: Frame) -> Frame:
        """Zero the excluded (masked-out) regions of the frame."""
        if isinstance(image, cv2.UMat):
            return cv2.bitwise_and(image, image, mask=self._roi_mask())
        image[self._mask] = 0  # host path writes in place
        return image

    def _morph_kernel(self) -> Frame:
        """Cached morphology kernel, rebuilt when the size changes."""
        kernel_size = self._morphology.kernel_size
        if self._kernel is None or self._kernel_size_cached != kernel_size:
            kernel: Frame = cv2.getStructuringElement(
                cv2.MORPH_RECT, (kernel_size, kernel_size)
            )
            if self._use_ocl:
                kernel = cv2.UMat(kernel)  # type: ignore[call-overload]
            self._kernel = kernel
            self._kernel_size_cached = kernel_size
        return self._kernel

    def _reset_state(self) -> None:
        """Restart warmup and persistence; the model keeps its background."""
        self._prev_boxes = []
        self._frame_idx = 0

    def _ptz_moving(self) -> bool:
        return (
            self.ptz_metrics is not None
            and self.ptz_metrics.autotracker_enabled.value
            and not self.ptz_metrics.motor_stopped.is_set()
        )

    def _normalize_contrast(self, image: Frame) -> Frame:
        """Rescale contrast with a LUT. The LUT stays on-device for UMat."""
        if self._contrast_rescale(self._histogram(image), self._proc_pixels):
            image = cv2.LUT(image, self._lut)  # type: ignore[call-overload]
        return image

    def _histogram(self, image: Frame) -> np.ndarray:
        """The 256-bin histogram, always returned on the host."""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        if isinstance(hist, cv2.UMat):
            hist = hist.get()
        return np.asarray(hist).flatten()

    def _contrast_rescale(self, hist: np.ndarray, total: int) -> bool:
        """Update the percentile window and self._lut. True means rescale now."""
        cum_hist = np.cumsum(hist)
        min_value = np.searchsorted(
            cum_hist, total * (self._contrast_min_pct / 100.0)
        ).astype(np.uint8)
        max_value = np.searchsorted(
            cum_hist, total * (self._contrast_max_pct / 100.0)
        ).astype(np.uint8)
        # one flat color: nothing to rescale
        if min_value >= max_value:
            return False
        self._contrast_values[self._contrast_index] = [min_value, max_value]
        self._contrast_index += 1
        if self._contrast_index == len(self._contrast_values):
            self._contrast_index = 0

        avg_min, avg_max = np.mean(self._contrast_values, axis=0)

        bins = np.arange(256)
        lut = np.clip((bins - avg_min) * (255.0 / (avg_max - avg_min + 1e-6)), 0, 255)
        self._lut = lut.astype(np.uint8)
        return True

    def _filter_persistent(
        self, boxes: list[tuple[int, int, int, int]]
    ) -> list[tuple[int, int, int, int]]:
        """Keep boxes seen for at least persistence_frames frames in a row.

        Boxes match across frames by center distance, so motion keeps its streak.
        """
        if self._persistence_frames <= 0:
            return boxes

        prev_boxes = [b for (b, _streak) in self._prev_boxes]
        prev_streaks = [s for (_box, s) in self._prev_boxes]

        # match each box to the nearest previous box
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
            # tolerance scales with box size
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
