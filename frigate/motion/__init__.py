from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from numpy import ndarray

from frigate.config import MotionConfig

if TYPE_CHECKING:
    from frigate.camera import PTZMetrics


class MotionDetector(ABC):
    @abstractmethod
    def __init__(
        self,
        frame_shape: tuple[int, int, int],
        config: MotionConfig,
        fps: int,
        improve_contrast: bool,
        threshold: int,
        contour_area: int | None,
    ) -> None:
        pass

    @abstractmethod
    def detect(self, frame: ndarray) -> list:
        """Detect motion and return motion boxes."""
        pass

    @abstractmethod
    def is_calibrating(self) -> bool:
        """Return if motion is recalibrating."""
        pass

    @abstractmethod
    def update_mask(self) -> None:
        """Update the motion mask after a config change."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop any ongoing work and processes."""
        pass


def create_motion_detector(
    frame_shape: tuple[int, int],
    config: MotionConfig,
    fps: int,
    name: str,
    ptz_metrics: PTZMetrics | None,
) -> MotionDetector:
    """Create the motion detector selected by the config detector field.

    Detector class imports stay inside the function so the default
    improved path never imports cv2_mog2_motion.
    """
    if getattr(config, "detector", "improved") == "mog2":
        from frigate.motion.cv2_mog2_motion import Cv2Mog2MotionDetector

        return Cv2Mog2MotionDetector(
            frame_shape, config, fps, name=name, ptz_metrics=ptz_metrics
        )

    from frigate.motion.improved_motion import ImprovedMotionDetector

    return ImprovedMotionDetector(
        frame_shape, config, fps, name=name, ptz_metrics=ptz_metrics
    )
