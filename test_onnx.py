import onnxruntime as ort
import imageio
import numpy as np


def load_image(path):
    return np.array(imageio.imread_v2(path)[..., :3].transpose(2, 0, 1), dtype=np.float32)[np.newaxis]

# Load the model and create InferenceSession
model_path = "weights/onnx/banet_1.onnx"
session = ort.InferenceSession(model_path)
# "Load and preprocess the input image inputTensor"
left = load_image("/home/quant/datasets/dsec/interlaken_00_c_images_rectified_left/000000.png")
right = load_image("/home/quant/datasets/dsec/interlaken_00_c_images_rectified_right/000000.png")

import models
from core.utils.utils import InputPadder
import torch
padder = InputPadder(left.shape, divis_by=32)
left, right = padder.pad(torch.from_numpy(left), torch.from_numpy(right))

# Run inference
outputs = session.run(None, {"left_image": left.numpy(), "right_image": right.numpy()})
print(outputs[0][0][0])

imageio.imwrite("out3.png", np.array(outputs[0][0][0], dtype=np.uint8))


