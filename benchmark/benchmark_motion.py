import cv2
from line_profiler import LineProfiler

from frigate.camera import PTZMetrics
from frigate.config.config import RuntimeMotionConfig
from frigate.motion.improved_motion import ImprovedMotionDetector
from frigate.util.config import get_relative_coordinates

cap = cv2.VideoCapture("/opt/frigate-dev/benchmark/peanut-ringo.m4v")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_shape = (height, width, 3)

mask = get_relative_coordinates("0.164,0.048,0.164,0.01,0.008,0.01,0.008,0.046", (height, width))

motion_config_1 = RuntimeMotionConfig(mask=mask)
motion_config_1.frame_height = 150

motion_config_2 = RuntimeMotionConfig(mask=mask)
motion_config_2.frame_height = 150
motion_config_2.threshold = 20

ptz = PTZMetrics(autotracker_enabled=False)

detector_1 = ImprovedMotionDetector(
    frame_shape=frame_shape,
    config=motion_config_1,
    fps=fps,
    ptz_metrics=ptz,
    name="default",
)
detector_1.save_images = False

detector_2 = ImprovedMotionDetector(
    frame_shape=frame_shape,
    config=motion_config_2,
    fps=fps,
    ptz_metrics=ptz,
    name="compare",
)
detector_2.save_images = False

lp = LineProfiler()
detector_1.detect = lp(detector_1.detect)
detector_2.detect = lp(detector_2.detect)

ret, frame = cap.read()
while ret:
    yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
    detector_1.detect(yuv_frame)
    detector_2.detect(yuv_frame)
    ret, frame = cap.read()

cap.release()
lp.print_stats()
