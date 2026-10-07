from typing import Any, Literal

from pydantic import Field, field_serializer

from ..base import FrigateBaseModel
from .mask import MotionMaskConfig

__all__ = [
    "Mog2ContoursConfig",
    "Mog2MorphologyConfig",
    "Mog2MotionConfig",
    "MotionConfig",
]


class Mog2MorphologyConfig(FrigateBaseModel):
    enabled: bool = Field(
        default=True,
        title="Enable morphology",
        description="Apply open/close morphology to the foreground mask to scrub speckle.",
    )
    kernel_size: int = Field(
        default=3,
        title="Morphology kernel size",
        description="Size of the square morphology kernel in pixels.",
        ge=1,
    )
    iterations: int = Field(
        default=3,
        title="Morphology iterations",
        description="Number of open/close morphology iterations.",
        ge=0,
    )


class Mog2ContoursConfig(FrigateBaseModel):
    min_area: int | None = Field(
        default=160,
        title="Minimum contour area",
        description="Minimum ROI area in pixels at the MOG2 processing resolution (tuned default 160 at the 360 processing height); unset (null) falls back to the contour area setting.",
        ge=0,
    )
    max_area_ratio: float = Field(
        default=0.9,
        title="Maximum contour area ratio",
        description="Maximum ROI area as a fraction of the motion frame; larger contours (e.g. full-frame scene changes) are ignored.",
        ge=0.0,
        le=1.0,
    )


class Mog2MotionConfig(FrigateBaseModel):
    frame_height: int | None = Field(
        default=None,
        title="Processing frame height",
        description="Height in pixels to scale frames to for MOG2 processing; unset uses 360 (tuned for MOG2). The stock motion frame_height (default 100) is too coarse for the mog2 detector and is not used by it.",
        ge=1,
    )
    history: int = Field(
        default=100,
        title="Background history",
        description="MOG2 background model history in frames; higher is more robust but uses more memory.",
        ge=1,
    )
    var_threshold: int = Field(
        default=24,
        title="Variance threshold",
        description="MOG2 variance threshold; higher values flag fewer pixels as foreground (less sensitive).",
        ge=1,
    )
    learning_rate: float | None = Field(
        default=None,
        title="Learning rate",
        description="Per-frame background adaptation rate in steady state (0 to 1). Leave unset to let MOG2 choose its automatic adaptive rate; a fixed low rate absorbs stationary objects into the background quickly and measurably reduces motion recall.",
        ge=0.0,
        le=1.0,
    )
    calibration_learning_rate: float | None = Field(
        default=None,
        title="Calibration learning rate",
        description="Background adaptation rate used while calibrating (startup, mask or option changes, lightning or skip triggers). Leave unset to let MOG2 choose its automatic adaptive rate.",
        ge=0.0,
        le=1.0,
    )
    shadow_mode: Literal["keep", "background"] = Field(
        default="keep",
        title="Shadow mode",
        description="Whether MOG2 shadow pixels count as foreground (keep) or are treated as background (background).",
    )
    use_bgr: bool = Field(
        default=False,
        title="Use BGR input",
        description=(
            "Convert the YUV420p (I420) frame to BGR and feed 3-channel color to "
            "MOG2 instead of the grayscale luma plane. Luma (default) is cheaper "
            "and cleaner on dappled shadows; BGR enables color-aware modeling "
            "but adds ~2x dappled false positives."
        ),
    )
    contrast_norm: bool = Field(
        default=True,
        title="Contrast normalization",
        description="Apply percentile contrast normalization before analysis to stabilize detection across lighting changes.",
    )
    contrast_history: int = Field(
        default=50,
        title="Contrast history",
        description="Moving window length in frames for the min/max percentile contrast baseline.",
        ge=1,
    )
    contrast_min_pct: float = Field(
        default=4.0,
        title="Contrast min percentile",
        description="Lower percentile bound for contrast normalization (0 to 100).",
        ge=0.0,
        le=100.0,
    )
    contrast_max_pct: float = Field(
        default=96.0,
        title="Contrast max percentile",
        description="Upper percentile bound for contrast normalization (0 to 100).",
        ge=0.0,
        le=100.0,
    )
    persistence_frames: int = Field(
        default=2,
        title="Persistence frames",
        description="Minimum consecutive frames a region must persist before a motion box is emitted; 0 disables the temporal gate.",
        ge=0,
    )
    persistence_match_tolerance: float = Field(
        default=0.5,
        title="Persistence match tolerance",
        description="Center distance match factor (times box size) used to track a region across frames for the persistence gate.",
        gt=0.0,
    )
    morphology: Mog2MorphologyConfig = Field(
        default_factory=Mog2MorphologyConfig,
        title="Morphology",
        description="Morphology options applied to the MOG2 foreground mask.",
    )
    contours: Mog2ContoursConfig = Field(
        default_factory=Mog2ContoursConfig,
        title="Contours",
        description="Contour area gate options for MOG2 motion boxes.",
    )
    warmup_frames: int = Field(
        default=30,
        title="Warmup frames",
        description="Frames the MOG2 model learns the background before emitting boxes.",
        ge=0,
    )


class MotionConfig(FrigateBaseModel):
    enabled: bool = Field(
        default=True,
        title="Enable motion detection",
        description="Enable or disable motion detection for all cameras; can be overridden per-camera.",
    )
    detector: Literal["improved", "mog2"] = Field(
        default="improved",
        title="Motion detector",
        description="Motion detector algorithm: improved is the stock detector (default) and mog2 is the OpenCV MOG2 background subtraction detector. Changing this requires a restart; the mog2 options are hot-reloadable. When using mog2, leave motion.frame_height unset (the detector defaults to its tuned 360 processing height; values below 200 degrade detection and log a warning).",
    )
    threshold: int = Field(
        default=30,
        title="Motion threshold",
        description="Pixel difference threshold used by the motion detector; higher values reduce sensitivity (range 1-255).",
        ge=1,
        le=255,
    )
    lightning_threshold: float = Field(
        default=0.8,
        title="Lightning threshold",
        description="Threshold to detect and ignore brief lighting spikes (lower is more sensitive, values between 0.3 and 1.0). This does not prevent motion detection entirely; it merely causes the detector to stop analyzing additional frames once the threshold is exceeded. Motion-based recordings are still created during these events.",
        ge=0.3,
        le=1.0,
    )
    skip_motion_threshold: float | None = Field(
        default=None,
        title="Skip motion threshold",
        description="If set to a value between 0.0 and 1.0, and more than this fraction of the image changes in a single frame, the detector will return no motion boxes and immediately recalibrate. This can save CPU and reduce false positives during lightning, storms, etc., but may miss real events such as a PTZ camera auto‑tracking an object. The trade‑off is between dropping a few megabytes of recordings versus reviewing a couple short clips. Leave unset (None) to disable this feature.",
        ge=0.0,
        le=1.0,
    )
    improve_contrast: bool = Field(
        default=True,
        title="Improve contrast",
        description="Apply contrast improvement to frames before motion analysis to help detection.",
    )
    contour_area: int | None = Field(
        default=10,
        title="Contour area",
        description="Minimum contour area in pixels required for a motion contour to be counted.",
    )
    delta_alpha: float = Field(
        default=0.2,
        title="Delta alpha",
        description="Alpha blending factor used in frame differencing for motion calculation.",
    )
    frame_alpha: float = Field(
        default=0.01,
        title="Frame alpha",
        description="Alpha value used when blending frames for motion preprocessing.",
    )
    frame_height: int | None = Field(
        default=100,
        title="Frame height",
        description="Height in pixels to scale frames to when computing motion.",
    )
    mask: dict[str, MotionMaskConfig | None] = Field(
        default_factory=dict,
        title="Mask coordinates",
        description="Ordered x,y coordinates defining the motion mask polygon used to include/exclude areas.",
    )
    mqtt_off_delay: int = Field(
        default=30,
        title="MQTT off delay",
        description="Seconds to wait after last motion before publishing an MQTT 'off' state.",
    )
    mog2: Mog2MotionConfig = Field(
        default_factory=Mog2MotionConfig,
        title="MOG2 options",
        description="Options for the MOG2 motion detector; only used when the detector is set to mog2.",
    )
    enabled_in_config: bool | None = Field(
        default=None,
        title="Original motion state",
        description="Indicates whether motion detection was enabled in the original static configuration.",
    )
    raw_mask: dict[str, MotionMaskConfig | None] = Field(
        default_factory=dict, exclude=True
    )

    @field_serializer("mask", when_used="json")
    def serialize_mask(self, value: Any, info):
        if self.raw_mask:
            return self.raw_mask
        return value

    @field_serializer("raw_mask", when_used="json")
    def serialize_raw_mask(self, value: Any, info):
        return None
