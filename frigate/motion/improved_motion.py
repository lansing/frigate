import logging

import cv2
import numpy as np
from line_profiler import profile
from scipy.ndimage import gaussian_filter

from frigate.camera import PTZMetrics
from frigate.config.config import RuntimeMotionConfig
from frigate.motion import MotionDetector
from frigate.util.image import grab_cv2_contours

logger = logging.getLogger(__name__)


class ImprovedMotionDetector(MotionDetector):
    def __init__(
        self,
        frame_shape: tuple[int, ...],
        config: RuntimeMotionConfig,
        fps: int,
        ptz_metrics: PTZMetrics | None = None,
        name: str = "improved",
        blur_radius: int = 1,
        interpolation: int = cv2.INTER_NEAREST,
        contrast_frame_history: int = 50,
    ) -> None:
        self.name = name
        self.config = config
        self.frame_shape = frame_shape
        frame_height = config.frame_height or frame_shape[0]
        self.resize_factor = frame_shape[0] / frame_height
        self.motion_frame_size = (
            frame_height,
            frame_height * frame_shape[1] // frame_shape[0],
        )
        self.avg_frame = np.zeros(self.motion_frame_size, np.float32)
        self.motion_frame_count = 0
        self.frame_counter = 0
        self.update_mask()
        self.save_images = False
        self.calibrating = True
        self.blur_radius = blur_radius
        self.interpolation = interpolation
        self.contrast_values = np.zeros((contrast_frame_history, 2), np.uint8)
        self.contrast_values[:, 1:2] = 255
        self.contrast_values_index = 0
        self.ptz_metrics = ptz_metrics
        self.last_stop_time: float | None = None

        self.lut = np.zeros((256,), dtype=np.uint8)

    def is_calibrating(self) -> bool:
        return self.calibrating

    @profile
    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        motion_boxes: list[tuple[int, int, int, int]] = []

        if not self.config.enabled:
            return motion_boxes

        # if ptz motor is moving from autotracking, quickly return
        # a single box that is 80% of the frame
        if self.ptz_metrics is not None and (
            self.ptz_metrics.autotracker_enabled.value
            and not self.ptz_metrics.motor_stopped.is_set()
        ):
            return [
                (
                    int(self.frame_shape[1] * 0.1),
                    int(self.frame_shape[0] * 0.1),
                    int(self.frame_shape[1] * 0.9),
                    int(self.frame_shape[0] * 0.9),
                )
            ]

        gray = frame[0 : self.frame_shape[0], 0 : self.frame_shape[1]]

        # resize frame
        resized_frame = cv2.resize(
            gray,
            dsize=(self.motion_frame_size[1], self.motion_frame_size[0]),
            interpolation=self.interpolation,
        )

        if self.save_images:
            resized_saved = resized_frame.copy()

        # Improve contrast
        if self.config.improve_contrast:
            # TODO tracking moving average of min/max to avoid sudden contrast changes
            # TODO SLOW!!
            # TODO 30% of time
            # min_value_old = np.percentile(resized_frame, 4).astype(np.uint8)
            # TODO 16% of time
            # max_value_old = np.percentile(resized_frame, 96).astype(np.uint8)
            # skip contrast calcs if the image is a single color
            # NOTE optimization 1: percentile_via_histogram instead of np.percentile usage
            min_value, max_value = self.percentile_via_histogram(resized_frame)
            # print(f"{min_value_old}->{min_value} {max_value_old}->{max_value}")
            if min_value < max_value:
                # keep track of the last 50 contrast values
                self.contrast_values[self.contrast_values_index] = [
                    min_value,
                    max_value,
                ]
                self.contrast_values_index += 1
                if self.contrast_values_index == len(self.contrast_values):
                    self.contrast_values_index = 0

                avg_min, avg_max = np.mean(self.contrast_values, axis=0)

                # resized_frame = np.clip(resized_frame, avg_min, avg_max)
                # resized_frame = (
                #     ((resized_frame - avg_min) / (avg_max - avg_min)) * 255
                # ).astype(np.uint8)

                # NOTE optimization 2 that we want to keep
                # This replaces both the np.clip and the (val - min) / (max - min) math
                # TODO refactor this into two lines. no need to initialize bins, probably no need to assign self.lut.
                bins = np.arange(256)
                lut_values = np.clip(
                    (bins - avg_min) * (255.0 / (avg_max - avg_min + 1e-6)), 0, 255
                )
                # TODO why is this an instance variable? just pass this to the next call inline if possible?
                self.lut = lut_values.astype(np.uint8)
                resized_frame = cv2.LUT(resized_frame, self.lut)

        if self.save_images:
            contrasted_saved = resized_frame.copy()

        # mask frame
        # this has to come after contrast improvement
        # Setting masked pixels to zero, to match the average frame at startup
        # TODO SLOW!
        # TODO 10%: 834     251555.7    301.6     15.9          resized_frame[self.mask] = [0]
        # resized_frame[self.mask] = [0]
        #
        # 1. Create a blank "keep all" mask (255 = white)
        # Use the motion frame dimensions: (150, 266)

        # print(f"mask: {len(self.mask)}")
        # for m in self.mask:
        # print(m.shape)
        # print(f"inv_mask:")
        # print(self.inv_mask.shape)
        # print(f"resized_frame:")
        # print(resized_frame.shape)
        # NOTE optimization 3 we want to keep
        resized_frame = cv2.bitwise_and(
            resized_frame, resized_frame, mask=self.inv_mask
        )

        # TODO SLOW
        # TODO 16%
        # resized_frame = gaussian_filter(resized_frame, sigma=1, radius=self.blur_radius)
        # NOTE optimization 4 we want to keep
        resized_frame = self.gaussian_via_cv2(resized_frame)

        if self.save_images:
            blurred_saved = resized_frame.copy()

        if self.save_images or self.calibrating:
            self.frame_counter += 1
        # compare to average
        frameDelta = cv2.absdiff(resized_frame, cv2.convertScaleAbs(self.avg_frame))

        # compute the threshold image for the current frame
        thresh = cv2.threshold(
            frameDelta, self.config.threshold, 255, cv2.THRESH_BINARY
        )[1]

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh_dilated = cv2.dilate(thresh, None, iterations=1)  # type: ignore[call-overload]
        contours = cv2.findContours(
            thresh_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = grab_cv2_contours(contours)

        # loop over the contours
        total_contour_area: float = 0
        for c in contours:
            # if the contour is big enough, count it as motion
            contour_area = cv2.contourArea(c)
            total_contour_area += contour_area
            if contour_area > (self.config.contour_area or 0):
                x, y, w, h = cv2.boundingRect(c)
                motion_boxes.append(
                    (
                        int(x * self.resize_factor),
                        int(y * self.resize_factor),
                        int((x + w) * self.resize_factor),
                        int((y + h) * self.resize_factor),
                    )
                )

        pct_motion = total_contour_area / (
            self.motion_frame_size[0] * self.motion_frame_size[1]
        )

        # check if the motor has just stopped from autotracking
        # if so, reassign the average to the current frame so we begin with a new baseline
        if self.ptz_metrics is not None and (
            # ensure we only do this for cameras with autotracking enabled
            self.ptz_metrics.autotracker_enabled.value
            and self.ptz_metrics.motor_stopped.is_set()
            and (
                self.last_stop_time is None
                or self.ptz_metrics.stop_time.value != self.last_stop_time
            )
            # value is 0 on startup or when motor is moving
            and self.ptz_metrics.stop_time.value != 0
        ):
            self.last_stop_time = self.ptz_metrics.stop_time.value

            self.avg_frame = resized_frame.astype(np.float32)
            motion_boxes = []
            pct_motion = 0

        # skip motion entirely if the scene change percentage exceeds configured
        # threshold. this is useful to ignore lighting storms, IR mode switches,
        # etc. rather than registering them as brief motion and then recalibrating.
        # note: skipping means the frame is dropped and **no recording will be
        # created**, which could hide a legitimate object if the camera is actively
        # auto‑tracking. the alternative is to allow motion and accept a small
        # recording that can be reviewed in the timeline. disabled by default (None).
        if (
            self.config.skip_motion_threshold is not None
            and pct_motion > self.config.skip_motion_threshold
        ):
            # force a recalibration so we transition to the new background
            self.calibrating = True
            return []

        # once the motion is less than 5% and the number of contours is < 4, assume its calibrated
        if pct_motion < 0.05 and len(motion_boxes) <= 4:
            self.calibrating = False

        # if calibrating or the motion contours are > 80% of the image area
        # (lightning, ir, ptz) recalibrate. the lightning threshold does **not**
        # stop motion detection entirely; it simply halts additional processing for
        # the current frame once the percentage crosses the threshold. this helps
        # reduce false positive object detections and CPU usage during high‑motion
        # events. recordings continue to be generated because users expect data
        # while a PTZ camera is moving.
        if self.calibrating or pct_motion > self.config.lightning_threshold:
            self.calibrating = True

        if self.save_images:
            thresh_dilated = cv2.cvtColor(thresh_dilated, cv2.COLOR_GRAY2BGR)
            for b in motion_boxes:
                cv2.rectangle(
                    thresh_dilated,
                    (int(b[0] / self.resize_factor), int(b[1] / self.resize_factor)),
                    (int(b[2] / self.resize_factor), int(b[3] / self.resize_factor)),
                    (0, 0, 255),
                    2,
                )
            frames = [
                cv2.cvtColor(resized_saved, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(contrasted_saved, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(blurred_saved, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(frameDelta, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR),
                thresh_dilated,
            ]
            cv2.imwrite(
                f"debug/frames/{self.name}-{self.frame_counter}.jpg",
                (
                    cv2.hconcat(frames)
                    if self.frame_shape[0] > self.frame_shape[1]
                    else cv2.vconcat(frames)
                ),
            )

        if len(motion_boxes) > 0:
            self.motion_frame_count += 1
            if self.motion_frame_count >= 10:
                # only average in the current frame if the difference persists for a bit
                cv2.accumulateWeighted(
                    resized_frame,
                    self.avg_frame,
                    0.2 if self.calibrating else self.config.frame_alpha,
                )
        else:
            # when no motion, just keep averaging the frames together
            cv2.accumulateWeighted(
                resized_frame,
                self.avg_frame,
                0.2 if self.calibrating else self.config.frame_alpha,
            )
            self.motion_frame_count = 0

        return motion_boxes

    @profile
    def gaussian_via_cv2(self, resized_frame):

        # Note: cv2.
        # GaussianBlur takes a kernel size tuple (width, height) which must be odd numbers.
        # If you used sigma=1, a standard kernel size is 5x5 or 7x7.
        # You'll need to translate your `self.blur_radius` to an odd integer kernel size.
        k_size = int(self.blur_radius * 2 + 1)  # Example conversion
        if k_size % 2 == 0:
            k_size += 1

        resized_frame = cv2.GaussianBlur(resized_frame, (k_size, k_size), sigmaX=1)
        return resized_frame

    @profile
    def percentile_via_histogram(self, resized_frame):
        # 1. Calculate the histogram using OpenCV (very fast)
        hist = cv2.calcHist([resized_frame], [0], None, [256], [0, 256]).flatten()

        # 2. Get the cumulative sum of the histogram
        cum_hist = np.cumsum(hist)

        # 3. Calculate the threshold index based on pixel count
        total_pixels = resized_frame.size
        min_thresh = total_pixels * 0.04
        max_thresh = total_pixels * 0.96

        # 4. Find the first intensity bin that crosses the thresholds
        min_value = np.searchsorted(cum_hist, min_thresh).astype(np.uint8)
        max_value = np.searchsorted(cum_hist, max_thresh).astype(np.uint8)
        return min_value, max_value

    def update_mask(self) -> None:
        resized_mask = cv2.resize(
            self.config.rasterized_mask,
            dsize=(self.motion_frame_size[1], self.motion_frame_size[0]),
            interpolation=cv2.INTER_AREA,
        )
        self.mask = np.where(resized_mask == [0])

        self.inv_mask = np.full(
            (self.motion_frame_size[0], self.motion_frame_size[1]), 255, dtype=np.uint8
        )
        self.inv_mask[self.mask] = 0

        # Reset motion detection state when mask changes
        # so motion detection can quickly recalibrate with the new mask
        self.avg_frame = np.zeros(self.motion_frame_size, np.float32)
        self.calibrating = True
        self.motion_frame_count = 0

    def stop(self) -> None:
        """stop the motion detector."""
        pass
