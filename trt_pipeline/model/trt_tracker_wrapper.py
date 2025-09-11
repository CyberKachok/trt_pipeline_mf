import pycuda.driver as cuda
import tensorrt as trt
import numpy as np

from mixformer_utils.processing_utils import Preprocessor_trt, sample_target, clip_box
from .tracker_wrapper import TrackerWrapper
import collections
import time

class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem


class TrtTrackerWrapper(TrackerWrapper):
    def __init__(self, cfg_path, engine_path):
        super().__init__(cfg_path)

        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine:
            print('[INFO] Engine deserialization done.')

        self.context = self.engine.create_execution_context()

        # Allocate buffers
        ctx = cuda.Context.attach()
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
        ctx.detach()

        # Set tensor address
        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])

        self.preprocessor = Preprocessor_trt()
        self.name = 'trt'
        self.fps_history = collections.deque(maxlen=10)


    def infer(self):
        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.outputs[0].host, self.outputs[0].device, self.stream)
        cuda.memcpy_dtoh_async(self.outputs[1].host, self.outputs[1].device, self.stream)

        # Synchronize the stream
        self.stream.synchronize()

        return self.outputs[0].host, self.outputs[1].host

    def initialize(self, image, init_bbox: list):
        # forward the template once
        z_patch_arr, _, = sample_target(image, init_bbox, 
                                        self.template_factor,
                                        output_sz=self.template_size)
        self.template = self.preprocessor.process(z_patch_arr)

        np.copyto(self.inputs[0].host, self.template.ravel())
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)

        np.copyto(self.inputs[1].host, self.template.ravel())
        cuda.memcpy_htod_async(self.inputs[1].device, self.inputs[1].host, self.stream)

        # save states
        self.state = init_bbox

    def track(self, image, frame_id: int):
        H, W, _ = image.shape
        x_patch_arr, resize_factor = sample_target(image, self.state, 
                                                   self.search_factor,
                                                   output_sz=self.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)

        np.copyto(self.inputs[2].host, search.ravel())
        cuda.memcpy_htod_async(self.inputs[2].device, self.inputs[2].host, self.stream)

        
        
        
        start_time = time.time()
        pred_boxes, pred_score = self.infer()
        per_time = time.time() - start_time
        self.fps_history.append(int((1/per_time)))
        avg_fps = sum(self.fps_history) / len(self.fps_history)



        pred_score = pred_score[0]
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes * self.search_size / resize_factor)  # (cx, cy, w, h) [0,1]

        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), 
                              H, W, margin=10)
        # update template

        if pred_score > 0.5 and pred_score > self.max_pred_score:
            z_patch_arr, _ = sample_target(image, self.state,
                                           self.template_factor,
                                           output_sz=self.template_size)  # (x1, y1, w, h)
            self.online_max_template = self.preprocessor.process(z_patch_arr)
            self.max_pred_score = pred_score
        if frame_id % self.update_interval == 0:
            np.copyto(self.inputs[1].host, self.online_max_template.ravel())
            cuda.memcpy_htod_async(self.inputs[1].device, self.inputs[1].host, self.stream)

            self.max_pred_score = -1
            self.online_max_template = self.template

        return self.state, pred_score, avg_fps


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []

    stream = cuda.Stream()
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        size = trt.volume(engine.get_tensor_shape(tensor_name))
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # # Append the device buffer address to device bindings
        bindings.append(int(device_mem))

        # # Append to the appropriate input/output list
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream
