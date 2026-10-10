import sys
import unittest
from unittest import mock

import cv2
import numpy as np

from frigate.config.camera.motion import Mog2ContoursConfig, MotionConfig
from frigate.motion import create_motion_detector
from frigate.motion.cv2_mog2_motion import Cv2Mog2MotionDetector
from frigate.motion.improved_motion import ImprovedMotionDetector


def make_config(frame_shape=(100, 100)) -> MotionConfig:
    """Build a MotionConfig with a full-include rasterized mask."""
    config = MotionConfig()
    object.__setattr__(
        config,
        "rasterized_mask",
        np.full((frame_shape[0], frame_shape[1]), 255, dtype=np.uint8),
    )
    return config


def shadow_scene() -> np.ndarray:
    """Static 128 frame with a soft-edged dark patch (~55% of background)."""
    frame = np.full((100, 100), 128, dtype=np.uint8)
    yy, xx = np.mgrid[0:40, 0:40]
    distance = np.sqrt((yy - 20) ** 2 + (xx - 20) ** 2) / 20.0
    values = (128 - 58 * np.clip(1 - distance, 0, 1)).astype(np.uint8)
    frame[30:70, 30:70] = values
    return frame


class TestCv2Mog2MotionDetector(unittest.TestCase):
    def setUp(self):
        self.frame_shape = (100, 100)
        self.config = make_config(self.frame_shape)
        # the default min_area (160) is tuned for the 360 processing height;
        # the 100x100 test frames need a smaller gate for their small objects
        self.config.mog2.contours.min_area = 10
        self.detector = Cv2Mog2MotionDetector(self.frame_shape, self.config, fps=30)

    def tearDown(self):
        # OCL tests toggle the process-global cv2 useOpenCL flag; keep the
        # state isolated between tests
        if getattr(cv2, "ocl", None) is not None:
            cv2.ocl.setUseOpenCL(False)

    def _static_frame(self) -> np.ndarray:
        return np.full((100, 100), 128, dtype=np.uint8)

    def _calibrate(self, detector=None, frames=60) -> None:
        """Feed static frames until the detector finishes calibrating."""
        detector = detector or self.detector
        for _ in range(frames):
            detector.detect(self._static_frame())
        self.assertFalse(detector.is_calibrating())

    def test_shadow_plane_present_in_grayscale(self):
        """MOG2 must emit a shadow plane (125/127) on grayscale input.

        Run first: the shadow_mode split (and its tests) depend on it.
        """
        sub = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=24, detectShadows=True
        )
        static = self._static_frame()
        for _ in range(60):
            sub.apply(static, 0.2)
        fg = sub.apply(shadow_scene(), 0.05)
        shadow_pixels = int(np.count_nonzero((fg > 0) & (fg < 255)))
        self.assertGreater(
            shadow_pixels,
            0,
            "expected a shadow plane (125/127) with grayscale input and "
            "detectShadows=True",
        )

    def test_static_scene_no_motion(self):
        """A constant scene emits no boxes and finishes calibrating."""
        boxes = None
        for _ in range(60):
            boxes = self.detector.detect(self._static_frame())
        self.assertEqual(boxes, [])
        self.assertFalse(self.detector.is_calibrating())

    def test_moving_object_emitted_after_persistence(self):
        """A moving object is tracked, delayed by the persistence gate, and
        continuous once established."""
        self._calibrate()

        boxes_by_frame = []
        for step in range(17):
            frame = self._static_frame()
            x = 5 + step * 5
            frame[45:55, x : x + 10] = 255
            boxes_by_frame.append(self.detector.detect(frame))

        first = next((i for i, boxes in enumerate(boxes_by_frame) if boxes), None)
        self.assertIsNotNone(first, "expected motion boxes for the moving object")
        # default persistence_frames=2: the box cannot appear on the first
        # motion frame
        self.assertGreaterEqual(first, 1)
        self.assertLessEqual(first, 3)

        remaining = boxes_by_frame[first:]
        box_frames = [boxes for boxes in remaining if boxes]
        self.assertGreaterEqual(
            len(box_frames),
            int(len(remaining) * 0.9),
            "expected continuous boxes once established",
        )

        for i in range(first, len(boxes_by_frame)):
            boxes = boxes_by_frame[i]
            if not boxes:
                continue
            object_cx = 5 + i * 5 + 5
            object_cy = 50
            matched = any(
                abs((box[0] + box[2]) / 2 - object_cx) <= 5
                and abs((box[1] + box[3]) / 2 - object_cy) <= 5
                for box in boxes
            )
            self.assertTrue(
                matched,
                f"frame {i}: no box center near the object: {boxes}",
            )

    def test_persistence_suppresses_flicker(self):
        """A patch that blinks on every other frame never reaches the
        persistence gate, while with the gate disabled it triggers."""
        config_off = make_config(self.frame_shape)
        config_off.mog2.persistence_frames = 0
        config_off.mog2.contours.min_area = 10
        det_off = Cv2Mog2MotionDetector(self.frame_shape, config_off, fps=30)
        det_on = Cv2Mog2MotionDetector(self.frame_shape, self.config, fps=30)

        static = self._static_frame()
        for det in (det_off, det_on):
            for _ in range(60):
                det.detect(static)

        off_triggers = 0
        on_triggers = 0
        for i in range(60):
            frame = static.copy()
            if i % 2 == 0:
                frame[20:29, 20:29] = 255
            if det_off.detect(frame):
                off_triggers += 1
            if det_on.detect(frame):
                on_triggers += 1

        self.assertGreater(
            off_triggers, 0, "flicker should trigger with persistence disabled"
        )
        self.assertEqual(
            on_triggers, 0, "flicker should not trigger with persistence enabled"
        )

    def test_shadow_mode_keep_and_background(self):
        """A shadow-like intensity drop is emitted in keep mode and ignored
        in background mode."""
        for mode, expect_box in (("keep", True), ("background", False)):
            with self.subTest(shadow_mode=mode):
                config = make_config(self.frame_shape)
                config.mog2.contrast_norm = False
                config.mog2.shadow_mode = mode
                det = Cv2Mog2MotionDetector(self.frame_shape, config, fps=30)

                static = self._static_frame()
                for _ in range(60):
                    det.detect(static)

                boxes = []
                # feed the scene a few frames so the persistence gate
                # (default 2) can pass the static patch through
                for _ in range(4):
                    boxes = det.detect(shadow_scene())

                if expect_box:
                    self.assertTrue(
                        boxes, "shadow_mode=keep should emit a box for the shadow"
                    )
                else:
                    self.assertEqual(
                        boxes,
                        [],
                        "shadow_mode=background should ignore the shadow",
                    )

    def test_lightning_recalibrates_without_rebuild(self):
        """A frame exceeding lightning_threshold recalibrates (warmup
        restarts) without rebuilding the MOG2 model."""
        self._calibrate()
        model_before = self.detector._sub

        lightning = self._static_frame()
        lightning[:90] = 200
        self.detector.detect(lightning)

        self.assertTrue(self.detector.is_calibrating())
        self.assertIs(
            self.detector._sub,
            model_before,
            "the MOG2 model must not be rebuilt on a lightning trigger",
        )

        for _ in range(self.config.mog2.warmup_frames):
            self.assertEqual(
                self.detector.detect(self._static_frame()),
                [],
                "warmup frames after a lightning trigger must emit nothing",
            )

    def test_config_surface(self):
        """persistence_frames=0 disables the gate; update_mask re-instantiates
        the model and resets state; stop() is safe."""
        config = make_config(self.frame_shape)
        config.mog2.persistence_frames = 0
        config.mog2.contours.min_area = 10
        det = Cv2Mog2MotionDetector(self.frame_shape, config, fps=30)
        static = self._static_frame()
        for _ in range(60):
            det.detect(static)

        frame = self._static_frame()
        frame[45:55, 45:55] = 255
        boxes = det.detect(frame)
        self.assertTrue(
            boxes, "persistence_frames=0 must emit on the first motion frame"
        )

        model_before = self.detector._sub
        self.config.mog2.var_threshold = 30
        self.detector.config = self.config
        self.detector.update_mask()
        self.assertIsNot(
            self.detector._sub,
            model_before,
            "update_mask must re-instantiate the MOG2 model",
        )
        self.assertTrue(self.detector.is_calibrating())
        self.assertEqual(self.detector._frame_idx, 0)
        self.assertEqual(self.detector._prev_boxes, [])

        self.detector.stop()

    def test_2d_and_3d_frames_equivalent(self):
        """A 2D luma frame and a 3D YUV-shaped frame with the same luma plane
        produce identical results."""
        config_2d = make_config(self.frame_shape)
        config_3d = make_config(self.frame_shape)
        config_2d.mog2.contours.min_area = 10
        config_3d.mog2.contours.min_area = 10
        det_2d = Cv2Mog2MotionDetector(self.frame_shape, config_2d, fps=30)
        det_3d = Cv2Mog2MotionDetector(self.frame_shape, config_3d, fps=30)

        results_2d = []
        results_3d = []
        for step in range(40):
            frame = self._static_frame()
            if step >= 10:
                x = (step - 10) * 3
                frame[45:55, x : x + 10] = 255
            yuv = np.zeros((150, 100), np.uint8)
            yuv[:100] = frame
            results_2d.append(det_2d.detect(frame))
            results_3d.append(det_3d.detect(yuv))

        self.assertEqual(results_2d, results_3d)

    def test_box_format_and_scale(self):
        """Boxes are full-frame (x1, y1, x2, y2) within the frame bounds and
        scaled by the resize factor."""
        config = make_config(self.frame_shape)
        config.mog2.frame_height = 50
        config.mog2.contours.min_area = 10
        det = Cv2Mog2MotionDetector(self.frame_shape, config, fps=30)
        static = self._static_frame()
        for _ in range(60):
            det.detect(static)

        all_boxes = []
        for step in range(11):
            frame = static.copy()
            x = 20 + step * 4
            frame[20:40, x : x + 20] = 255
            all_boxes.extend(det.detect(frame))

        self.assertTrue(all_boxes, "expected motion boxes for the moving object")
        for x1, y1, x2, y2 in all_boxes:
            self.assertGreater(x2, x1)
            self.assertGreater(y2, y1)
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, 100)
            self.assertLessEqual(y2, 100)

        # with frame_height=50 the proc-space box is scaled by 2x; the first
        # box must land near the object center
        first = all_boxes[0]
        center_x = (first[0] + first[2]) / 2
        center_y = (first[1] + first[3]) / 2
        self.assertAlmostEqual(center_x, 30, delta=6)
        self.assertAlmostEqual(center_y, 30, delta=6)

    def test_mog2_default_processing_height(self):
        """Without an explicit mog2.frame_height the detector uses the tuned
        default 360, capped at the native resolution for small cameras."""
        config = make_config((1080, 1920))
        det = Cv2Mog2MotionDetector((1080, 1920), config, fps=30)
        self.assertEqual(det._proc_size, (360, 640))
        self.assertEqual(det._resize_factor, 3.0)

        small_config = make_config((180, 320))
        small_det = Cv2Mog2MotionDetector((180, 320), small_config, fps=30)
        self.assertEqual(small_det._proc_size, (180, 320))

    def test_mog2_explicit_frame_height(self):
        """An explicit mog2.frame_height is honored."""
        config = make_config((1080, 1920))
        config.mog2.frame_height = 540
        det = Cv2Mog2MotionDetector((1080, 1920), config, fps=30)
        self.assertEqual(det._proc_size, (540, 960))
        self.assertAlmostEqual(det._resize_factor, 2.0)

    def test_mog2_warns_on_low_frame_height(self):
        """An explicit processing height below the recommended minimum logs a
        warning (once per distinct value)."""
        config = make_config((1080, 1920))
        config.mog2.frame_height = 100
        with self.assertLogs("frigate.motion.cv2_mog2_motion", level="WARNING") as cm:
            det = Cv2Mog2MotionDetector((1080, 1920), config, fps=30)
            # a rebuild with the same low height must not warn again
            det.update_mask()
        self.assertTrue(
            any("below the recommended" in message for message in cm.output)
        )
        self.assertEqual(len(cm.output), 1)
        self.assertEqual(det._proc_size, (100, 178))

    def test_mog2_min_area_default(self):
        """The tuned min_area default is 160 (at the 360 processing height)."""
        self.assertEqual(Mog2ContoursConfig().min_area, 160)

    def test_factory_selects_detector(self):
        """The factory returns the selected detector class and the improved
        path never imports cv2_mog2_motion."""
        config_mog2 = make_config(self.frame_shape)
        config_mog2.detector = "mog2"
        det = create_motion_detector(
            self.frame_shape, config_mog2, 30, name="test", ptz_metrics=None
        )
        self.assertIsInstance(det, Cv2Mog2MotionDetector)

        config_improved = make_config(self.frame_shape)
        saved = sys.modules.pop("frigate.motion.cv2_mog2_motion", None)
        try:
            det = create_motion_detector(
                self.frame_shape, config_improved, 30, name="test", ptz_metrics=None
            )
            self.assertIsInstance(det, ImprovedMotionDetector)
            self.assertNotIn("frigate.motion.cv2_mog2_motion", sys.modules)
        finally:
            if saved is not None:
                sys.modules["frigate.motion.cv2_mog2_motion"] = saved

    def test_opencl_disabled_without_platform(self):
        """No OpenCL platform: pure CPU path, detector works as before."""
        with mock.patch.object(cv2.ocl, "haveOpenCL", return_value=False):
            det = Cv2Mog2MotionDetector(self.frame_shape, self.config, fps=30)
        self.assertFalse(det._use_ocl)

        self._calibrate(detector=det)
        boxes = []
        for _ in range(3):
            frame = self._static_frame()
            frame[45:55, 20:30] = 255
            boxes = det.detect(frame)
        self.assertTrue(boxes, "expected motion boxes on the CPU path")

    def test_opencl_enabled_with_platform(self):
        """An OpenCL platform is probed greedily and detect() runs the fused
        UMat pipeline."""
        with mock.patch.object(cv2.ocl, "haveOpenCL", return_value=True):
            det = Cv2Mog2MotionDetector(self.frame_shape, self.config, fps=30)
        self.assertTrue(det._use_ocl)

        # the fused chain builds its cached device-side mask; the plain CPU
        # body of detect() never would, so this proves the dispatch happened
        det.detect(self._static_frame())
        self.assertIsNotNone(
            det._inv_mask_umat, "expected detect() to run the fused UMat pipeline"
        )

        self._calibrate(detector=det)
        self.assertTrue(det._use_ocl, "no fallback should occur on the happy path")

        boxes = []
        for _ in range(3):
            frame = self._static_frame()
            frame[45:55, 20:30] = 255
            boxes = det.detect(frame)
        self.assertTrue(boxes)

    def test_opencl_falls_back_to_cpu_on_apply_error(self):
        """A runtime OpenCL fault in the fused pipeline switches the detector
        to CPU permanently (the model rebuild inside the fallback replaces
        _sub, so later frames run on a clean CPU model)."""
        with mock.patch.object(cv2.ocl, "haveOpenCL", return_value=True):
            det = Cv2Mog2MotionDetector(self.frame_shape, self.config, fps=30)
        self.assertTrue(det._use_ocl)

        real_apply = cv2.BackgroundSubtractorMOG2.apply
        state = {"failed": False}

        def flaky_apply(self, *args, **kwargs):
            if not state["failed"]:
                state["failed"] = True
                raise cv2.error("simulated OpenCL failure")
            return real_apply(self, *args, **kwargs)

        with (
            mock.patch.object(cv2.BackgroundSubtractorMOG2, "apply", flaky_apply),
            self.assertLogs("frigate.motion.cv2_mog2_motion", level="WARNING") as cm,
        ):
            # the fused apply hits the fault; the CPU fallback apply for the
            # same frame delegates to the real method (state already raised)
            det.detect(self._static_frame())
        self.assertTrue(
            any("falling back to CPU" in message for message in cm.output),
            "expected the CPU fallback warning",
        )
        self.assertFalse(det._use_ocl)

        # the faulted frame's model is rebuilt and re-warmed: feed static
        # frames to recalibrate, then the CPU path must detect motion
        self._calibrate(detector=det)
        boxes = []
        for step in range(17):
            frame = self._static_frame()
            x = 5 + step * 5
            frame[45:55, x : x + 10] = 255
            boxes = det.detect(frame)
            if boxes:
                break
        self.assertTrue(boxes, "expected the CPU fallback path to detect motion")

    def test_detect_dispatches_to_fused_pipeline(self):
        """detect() hands the frame to detect_ocl() at the top while the GPU
        is enabled, and runs its own CPU body once it is disabled."""
        with mock.patch.object(cv2.ocl, "haveOpenCL", return_value=True):
            det = Cv2Mog2MotionDetector(self.frame_shape, self.config, fps=30)
        self.assertTrue(det._use_ocl)

        sentinel = [(1, 2, 3, 4)]
        det.detect_ocl = lambda frame: sentinel
        self.assertIs(det.detect(self._static_frame()), sentinel)

        det._use_ocl = False
        det.detect_ocl = lambda frame: sentinel
        self.assertIsNot(det.detect(self._static_frame()), sentinel)

    def test_detect_ocl_matches_detect(self):
        """The fused GPU pipeline emits exactly the same boxes as the CPU
        pipeline for an identical frame sequence (the UMat ops used by the
        fused path are bit-exact, and the MOG2 GPU output was verified
        identical to the CPU output)."""
        det_cpu = Cv2Mog2MotionDetector(self.frame_shape, self.config, fps=30)
        det_ocl = Cv2Mog2MotionDetector(self.frame_shape, self.config, fps=30)
        # force the fused chain so it runs even where no OpenCL platform
        # exists (UMat ops then execute CPU-backed)
        det_ocl._use_ocl = True

        results_cpu = []
        results_ocl = []
        for step in range(90):
            frame = self._static_frame()
            if step >= 40:
                x = 5 + (step - 40) * 5
                frame[45:55, x : x + 10] = 255
            results_cpu.append(det_cpu.detect(frame))
            results_ocl.append(det_ocl.detect_ocl(frame))

        self.assertEqual(results_cpu, results_ocl)
        self.assertTrue(any(results_cpu), "expected motion boxes for the object")

    def test_detect_ocl_use_bgr_fused_color_pipeline(self):
        """In use_bgr mode the fused path uploads the raw I420 buffer and
        converts to BGR on-device (UMat cvtColor); the result matches the
        CPU BGR pipeline for an identical frame sequence."""
        self.config.mog2.use_bgr = True
        det_cpu = Cv2Mog2MotionDetector(self.frame_shape, self.config, fps=30)
        det_ocl = Cv2Mog2MotionDetector(self.frame_shape, self.config, fps=30)
        # force the fused chain so it runs even where no OpenCL platform
        # exists (UMat ops then execute CPU-backed and bit-exact)
        det_ocl._use_ocl = True

        def i420_frame() -> np.ndarray:
            return np.full((150, 100), 128, np.uint8)

        results_cpu: list[tuple[int, int, int, int]] = []
        results_ocl: list[tuple[int, int, int, int]] = []
        for step in range(90):
            frame = i420_frame()
            if step >= 40:
                x = 5 + (step - 40) * 5
                frame[45:55, x : x + 10] = 255
            results_cpu.append(det_cpu.detect(frame))
            results_ocl.append(det_ocl.detect_ocl(frame))

        self.assertTrue(any(results_cpu), "expected motion boxes from the BGR pipeline")
        self.assertEqual(results_cpu, results_ocl)

    def test_detect_ocl_falls_back_on_gpu_error(self):
        """A runtime OpenCL fault in the fused pipeline switches the
        detector to the CPU path permanently and serves the frame."""
        with mock.patch.object(cv2.ocl, "haveOpenCL", return_value=True):
            det = Cv2Mog2MotionDetector(self.frame_shape, self.config, fps=30)
        self.assertTrue(det._use_ocl)

        # calibrate through the fused pipeline first
        for _ in range(60):
            det.detect_ocl(self._static_frame())
        self.assertFalse(det.is_calibrating())

        state = {"failed": False}

        def flaky_mask_umat():
            if not state["failed"]:
                state["failed"] = True
                raise cv2.error("simulated OpenCL failure")
            return cv2.UMat(det._inv_mask)

        det._get_inv_mask_umat = flaky_mask_umat

        with self.assertLogs("frigate.motion.cv2_mog2_motion", level="WARNING") as cm:
            frame = self._static_frame()
            frame[45:55, 20:30] = 255
            det.detect_ocl(frame)
        self.assertTrue(
            any("falling back to CPU" in message for message in cm.output),
            "expected the CPU fallback warning",
        )
        self.assertFalse(det._use_ocl)

        # the faulted frame's model is rebuilt and re-warmed: feed static
        # frames to recalibrate (via the fused path, now CPU-backed). Then
        # a moving object must be detected (a static one at the fault
        # location was learned into the background by the faulted frame's
        # own fallback apply)
        for _ in range(60):
            det.detect_ocl(self._static_frame())
        self.assertFalse(det.is_calibrating())
        boxes = []
        for step in range(6):
            frame = self._static_frame()
            x = 10 + step * 5
            frame[45:55, x : x + 10] = 255
            boxes = det.detect_ocl(frame)
        self.assertTrue(boxes, "expected motion boxes after the GPU fallback")

    def test_detect_ocl_umat_caches_invalidate_on_update_mask(self):
        """update_mask drops the cached device-side mask and kernel, and the
        kernel cache rebuilds at the new size."""
        self._calibrate()
        # force the fused chain so it runs even where no OpenCL platform
        # exists (UMat ops then execute CPU-backed)
        self.detector._use_ocl = True
        self.detector.detect_ocl(self._static_frame())
        self.assertIsNotNone(self.detector._inv_mask_umat)
        self.assertIsNotNone(self.detector._kernel_umat)

        self.config.mog2.morphology.kernel_size = 5
        self.detector.update_mask()
        self.assertIsNone(self.detector._inv_mask_umat)
        self.assertIsNone(self.detector._kernel_umat)

        # feed past the warmup window so the full pipeline (and the kernel
        # cache build) actually runs
        for _ in range(60):
            self.detector.detect_ocl(self._static_frame())
        self.assertIsNotNone(self.detector._inv_mask_umat)
        self.assertEqual(self.detector._kernel_umat.get().shape[:2], (5, 5))

    def test_use_bgr_builds_color_input(self):
        """use_bgr converts the I420 frame to 3-channel BGR (real color);
        the default feeds the 2-D luma plane, and the flag hot-reloads via
        update_mask."""
        i420 = np.full((150, 100), 128, np.uint8)
        i420[100:150, :] = 80  # offset chroma (rows below the Y plane)

        self.assertEqual(self.detector._build_input(i420).shape, (100, 100))

        self.config.mog2.use_bgr = True
        self.detector.update_mask()
        bgr = self.detector._build_input(i420)
        self.assertEqual(bgr.shape, (100, 100, 3))
        self.assertFalse(
            (bgr[..., 0] == bgr[..., 1]).all(),
            "expected color (non-gray) output from the offset chroma",
        )

    def test_use_bgr_detects_motion(self):
        """The 3-channel BGR pipeline detects a moving object end to end."""
        self.config.mog2.use_bgr = True
        det = Cv2Mog2MotionDetector(self.frame_shape, self.config, fps=30)

        def i420_frame() -> np.ndarray:
            return np.full((150, 100), 128, np.uint8)

        for _ in range(60):
            det.detect(i420_frame())
        self.assertFalse(det.is_calibrating())
        boxes = []
        for step in range(6):
            frame = i420_frame()
            x = 10 + step * 5
            frame[45:55, x : x + 10] = 255
            boxes = det.detect(frame)
        self.assertTrue(boxes, "expected motion boxes from the BGR pipeline")

    def test_use_bgr_skips_contrast_norm(self):
        """BGR mode skips the luma percentile contrast step; luma mode runs
        it (the step is tuned for the single-channel luma plane)."""

        # non-uniform frames so the contrast step has work to do in either
        # mode (a single-color frame would short-circuit inside it)
        def var_luma() -> np.ndarray:
            f = np.full((100, 100), 128, np.uint8)
            f[10:40, 10:40] = 60
            f[60:90, 60:90] = 190
            return f

        def var_i420() -> np.ndarray:
            f = np.full((150, 100), 128, np.uint8)
            f[10:40, 10:40] = 60
            f[60:90, 60:90] = 190
            return f

        # control: the luma pipeline invokes the contrast step
        luma = Cv2Mog2MotionDetector(self.frame_shape, self.config, fps=30)
        luma_calls: list[np.ndarray] = []
        luma._normalize_contrast = lambda small: (luma_calls.append(small), small)[1]
        for _ in range(5):
            luma.detect(var_luma())
        self.assertGreater(len(luma_calls), 0, "expected contrast to run in luma mode")

        # BGR mode: the contrast step is skipped entirely
        self.config.mog2.use_bgr = True
        bgr = Cv2Mog2MotionDetector(self.frame_shape, self.config, fps=30)
        bgr_calls: list[np.ndarray] = []
        bgr._normalize_contrast = lambda small: (bgr_calls.append(small), small)[1]
        for _ in range(5):
            bgr.detect(var_i420())
        self.assertEqual(bgr_calls, [], "expected contrast to be skipped in BGR mode")


if __name__ == "__main__":
    unittest.main()
