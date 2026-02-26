import ctypes
import logging

import numpy as np

try:
    import tensorrt as trt
    from cuda import cuda

    TRT_SUPPORT = True
except ModuleNotFoundError:
    TRT_SUPPORT = False

from pydantic import Field
from typing_extensions import Literal

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig, ModelTypeEnum
from frigate.util.model import (
    post_process_dfine,
    post_process_rfdetr,
    post_process_yolo,
    post_process_yolox,
)

logger = logging.getLogger(__name__)

DETECTOR_KEY = "tensorrt"

if TRT_SUPPORT:

    class TrtLogger(trt.ILogger):
        def log(self, severity, msg):
            logger.log(self.getSeverity(severity), msg)

        def getSeverity(self, sev: trt.ILogger.Severity) -> int:
            if sev == trt.ILogger.VERBOSE:
                return logging.DEBUG
            elif sev == trt.ILogger.INFO:
                return logging.INFO
            elif sev == trt.ILogger.WARNING:
                return logging.WARNING
            elif sev == trt.ILogger.ERROR:
                return logging.ERROR
            elif sev == trt.ILogger.INTERNAL_ERROR:
                return logging.CRITICAL
            else:
                return logging.DEBUG


class TensorRTDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    device: int = Field(default=0, title="GPU Device Index")


class HostDeviceMem:
    def __init__(self, size, dtype):
        self.size = size
        self.dtype = dtype
        self.nbytes = size * np.dtype(dtype).itemsize
        err, self.host_ptr = cuda.cuMemHostAlloc(
            self.nbytes, cuda.CU_MEMHOSTALLOC_DEVICEMAP
        )
        err, self.device_ptr = cuda.cuMemAlloc(self.nbytes)
        self.host = np.frombuffer(
            (ctypes.c_byte * self.nbytes).from_address(int(self.host_ptr)),
            dtype=self.dtype,
        )

    def __del__(self):
        cuda.cuMemFreeHost(self.host_ptr)
        cuda.cuMemFree(self.device_ptr)


class TensorRtDetector(DetectionApi):
    type_key = DETECTOR_KEY

    def _load_engine(self, model_path):
        try:
            trt.init_libnvinfer_plugins(self.trt_logger, "")
            ctypes.cdll.LoadLibrary("/usr/local/lib/libyolo_layer.so")
        except OSError as e:
            # TODO does this matter? it does postprocessing for classic yolos on gpu?
            logger.warning(
                "failed to load libraries. %s",
                e,
            )

        with open(model_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _get_input_shape(self):
        """Get input shape of the TensorRT engine."""
        name = self.engine.get_tensor_name(0)
        binding_dims = self.engine.get_tensor_shape(name)
        dtype = trt.nptype(self.engine.get_tensor_dtype(name))
        if len(binding_dims) == 4:
            return (tuple(binding_dims[2:]), dtype)
        elif len(binding_dims) == 3:
            return (tuple(binding_dims[1:]), dtype)
        else:
            raise ValueError("bad dims of binding %s: %s" % (name, str(binding_dims)))

    def _allocate_buffers(self):
        """Allocates all host/device in/out buffers required for an engine."""
        inputs = []
        outputs = []
        bindings = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            size = trt.volume(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            mem = HostDeviceMem(size, dtype)
            bindings.append(int(mem.device_ptr))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                logger.debug(
                    "Input tensor %s shape: %s",
                    name,
                    self.engine.get_tensor_shape(name),
                )
                inputs.append(mem)
            else:
                logger.debug(
                    "Output tensor %s shape: %s",
                    name,
                    self.engine.get_tensor_shape(name),
                )
                outputs.append(mem)
        return inputs, outputs, bindings

    def _do_inference(self):
        cuda.cuCtxPushCurrent(self.cu_ctx)

        for inp in self.inputs:
            cuda.cuMemcpyHtoDAsync(
                inp.device_ptr, inp.host_ptr, inp.nbytes, self.stream
            )

        if not self.context.execute_v2(self.bindings):
            logger.warning("execute_v2 returned false")

        for out in self.outputs:
            cuda.cuMemcpyDtoHAsync(
                out.host_ptr, out.device_ptr, out.nbytes, self.stream
            )

        cuda.cuStreamSynchronize(self.stream)
        cuda.cuCtxPopCurrent()

        return [
            out.host.reshape(
                self.engine.get_tensor_shape(
                    self.engine.get_tensor_name(len(self.inputs) + i)
                )
            )
            for i, out in enumerate(self.outputs)
        ]

    def __init__(self, detector_config: TensorRTDetectorConfig):
        assert (
            TRT_SUPPORT
        ), f"TensorRT libraries not found, {DETECTOR_KEY} detector not present"

        super().__init__(detector_config)

        (cuda_err,) = cuda.cuInit(0)
        assert (
            cuda_err == cuda.CUresult.CUDA_SUCCESS
        ), f"Failed to initialize cuda {cuda_err}"
        err, dev_count = cuda.cuDeviceGetCount()
        logger.debug(f"Num Available Devices: {dev_count}")
        assert (
            detector_config.device < dev_count
        ), f"Invalid TensorRT Device Config. Device {detector_config.device} Invalid."
        err, self.cu_ctx = cuda.cuCtxCreate(
            cuda.CUctx_flags.CU_CTX_MAP_HOST, detector_config.device
        )

        err, self.stream = cuda.cuStreamCreate(0)
        self.trt_logger = TrtLogger()
        self.engine = self._load_engine(detector_config.model.path)
        self.input_shape = self._get_input_shape()
        self.model_type = detector_config.model.model_type

        try:
            self.context = self.engine.create_execution_context()
            (
                self.inputs,
                self.outputs,
                self.bindings,
            ) = self._allocate_buffers()
        except Exception as e:
            logger.error(e)
            raise RuntimeError("fail to allocate CUDA resources") from e

        if self.model_type == ModelTypeEnum.yolox:
            self.calculate_grids_strides()

        logger.info(
            "TensorRT detector initialized. Congratulations on running TensorRT on this system!"
        )
        logger.debug("TensorRT loaded. Input shape is %s", self.input_shape)

    def __del__(self):
        """Free CUDA memories."""
        if hasattr(self, "outputs") and self.outputs is not None:
            del self.outputs
        if hasattr(self, "inputs") and self.inputs is not None:
            del self.inputs
        if hasattr(self, "stream") and self.stream is not None:
            cuda.cuStreamDestroy(self.stream)
            del self.stream
        if hasattr(self, "engine"):
            del self.engine
        if hasattr(self, "context"):
            del self.context
        if hasattr(self, "trt_logger"):
            del self.trt_logger
        if hasattr(self, "cu_ctx"):
            cuda.cuCtxDestroy(self.cu_ctx)

    def _postprocess_yolo(self, trt_outputs, conf_th):
        # TODO we should detect whether libyolo was loaded, and rely on libyolo in that case
        # see mainline tensorrt.py for what they do
        """Postprocess TensorRT outputs for legacy SSD/YOLO format.

        # Args
            trt_outputs: a list of 2 or 3 tensors, where each tensor
                        contains a multiple of 7 float32 numbers in
                        the order of [x, y, w, h, box_confidence, class_id, class_prob]
            conf_th: confidence threshold
        # Returns
            detections array of shape (20, 6)
        """
        detection_list = []
        for o in trt_outputs:
            detections = o.reshape((-1, 7))
            detections = detections[detections[:, 4] * detections[:, 6] >= conf_th]
            detection_list.append(detections)
        detection_list = np.concatenate(detection_list, axis=0)

        if len(detection_list) == 0:
            return np.zeros((20, 6), np.float32)

        detection_list[:, 4] = detection_list[:, 4] * detection_list[:, 6]
        ordered = detection_list[detection_list[:, 4].argsort()[::-1]][:, 0:6]
        ordered[:, 2] = np.clip(ordered[:, 2] + ordered[:, 0], 0, 1)
        ordered[:, 3] = np.clip(ordered[:, 3] + ordered[:, 1], 0, 1)
        detections = ordered[:, [5, 4, 1, 0, 3, 2]][:20]

        append_cnt = 20 - len(detections)
        if append_cnt > 0:
            detections = np.append(
                detections, np.zeros((append_cnt, 6), np.float32), axis=0
            )

        return detections

    def detect_raw(self, tensor_input):
        # Output tensor of float32 of shape [20, 6] where:
        # 0 - class id
        # 1 - score
        # 2..5 - a value between 0 and 1 of the box: [top, left, bottom, right]

        if self.model_type == ModelTypeEnum.dfine:
            np.copyto(self.inputs[0].host, tensor_input.ravel(), casting="unsafe")
            np.copyto(
                self.inputs[1].host,
                np.array([[self.height, self.width]], dtype=np.int64).ravel(),
            )
            trt_outputs = self._do_inference()
            return post_process_dfine(trt_outputs, self.width, self.height)

        np.copyto(self.inputs[0].host, tensor_input.ravel(), casting="unsafe")
        trt_outputs = self._do_inference()

        if self.model_type == ModelTypeEnum.rfdetr:
            return post_process_rfdetr(trt_outputs)
        elif self.model_type == ModelTypeEnum.yologeneric:
            return post_process_yolo(trt_outputs, self.width, self.height)
        elif self.model_type == ModelTypeEnum.yolox:
            return post_process_yolox(
                trt_outputs[0],
                self.width,
                self.height,
                self.grids,
                self.expanded_strides,
            )
        elif self.model_type == ModelTypeEnum.yolonas:
            predictions = trt_outputs[0]
            detections = np.zeros((20, 6), np.float32)
            for i, prediction in enumerate(predictions):
                if i == 20:
                    break
                (_, x_min, y_min, x_max, y_max, confidence, class_id) = prediction
                if class_id < 0:
                    break
                detections[i] = [
                    class_id,
                    confidence,
                    y_min / self.height,
                    x_min / self.width,
                    y_max / self.height,
                    x_max / self.width,
                ]
            return detections
        else:  # ssd / legacy YOLO TRT format
            return self._postprocess_yolo(trt_outputs, self.thresh)
