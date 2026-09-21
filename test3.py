# import h5py
# import hdf5plugin

# events = h5py.File("/home/quant/datasets/dsec/interlaken_00_c_events_left/events.h5")

# print(events["events"]["p"])

import imageio

import torch
import numpy as np
import cv2
import tqdm
import numpy as np
import time
from pathlib import Path
from torch.utils.flop_counter import FlopCounterMode

DEVICE = "cuda"

export_model = False

if export_model:
    DEVICE = "cpu"

import models
from core.utils.utils import InputPadder

class BanetWithPadder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.banet = models.load_banet(DEVICE)

    def forward(self, left, right):
        padder = InputPadder(left.shape, divis_by=32)
        imgs = padder.pad(left, right)
        return padder.unpad(self.banet(*imgs))
        

def convert(img):
    import torchvision
    # img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
    img_o = img
    #print(img.shape)
    img = img.astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE), img_o

def write_pointcloud(path, depthmap, orig_image):
    with open(path, "w") as f:
        pixels = depthmap.reshape((-1, 3))
        color_pixels = orig_image.reshape((-1, 3))
        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(pixels)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
        f.write(f"\n".join(header)+"\n")
        for pixel, color in zip(pixels, color_pixels):
            f.write(f"{pixel[0]:.2f} {pixel[1]:.2f} {pixel[2]:.2f} {int(color[0])} {int(color[1])} {int(color[2])}\n")

def do_export_model(banet, imgs):
    print("Exporting for shape", imgs[0].shape) #  torch.Size([1, 3, 416, 896])
    banet.eval()
    batch = torch.export.Dim("batch")
    width = torch.export.Dim("width")
    height = torch.export.Dim("height")
    torch.onnx.export(banet, imgs, "weights/onnx/banet_3.onnx", input_names=["left_image", "right_image"], dynamo=True, external_data=False, dynamic_shapes=[
        {0: batch, 2: height, 3: width},
        {0: batch, 2: height, 3: width},
    ])


#write_pointcloud("model_results/test.ply", np.ones((640, 480))*0.5)


# left_imgs = Path("/home/quant/datasets/drivingstereo/rainy/left-image-half-size/")
# right_imgs = Path("/home/quant/datasets/drivingstereo/rainy/right-image-half-size/")
left_imgs = Path("/home/quant/datasets/dsec/interlaken_00_c_images_rectified_left")
right_imgs = Path("/home/quant/datasets/dsec/interlaken_00_c_images_rectified_right")
# left_imgs = Path("/home/quant/datasets/dsec/zurich_city_03_a_images_rectified_left")
# right_imgs = Path("/home/quant/datasets/dsec/zurich_city_03_a_images_rectified_right")

frame_id = 0

if True:
    disparity_matrix = np.array([
        [1.0, 0.0, 0.0, -713.5791168212891],
        [0.0, 1.0, 0.0, -570.9349365234375],
        [0.0, 0.0, 0.0, 1164.6238115833075],
        [0.0, 0.0, 1.9625989469856626, -0.0],
        ])
else:
    disparity_matrix = np.array([
        [1.0, 0.0, 0.0, -724.4121398925781],
        [0.0, 1.0, 0.0, -569.1058044433594],
        [0.0, 0.0, 0.0, 1150.8249465165975],
        [0.0, 0.0, 1.9640233116065924, -0.0],
        ])

banet = BanetWithPadder()

for path_left in tqdm.tqdm(sorted(list(left_imgs.glob("*.png")) + list(left_imgs.glob("*.jpg")))):
    if frame_id % 60 != 0:
        frame_id += 1
        continue
    
    path_right = right_imgs / path_left.name
    #print(path_left, path_right)

    imgs = (convert(imageio.imread_v2(path_left)), convert(imageio.imread_v2(path_right)))
    img_o = imgs[0][1]
    imageio.imwrite(f"model_results/banet-{frame_id:06d}-orig.png", img_o)
    imgs = imgs[0][0], imgs[1][0]

    with torch.inference_mode():
        if export_model:
            do_export_model(banet, tuple(imgs))
            break
        start = time.monotonic()
        with FlopCounterMode() as flop_counter:
            disparity = banet(*imgs).cpu().numpy().squeeze()
            total_flops = flop_counter.get_total_flops()
            print("total_flops", total_flops)
        elapsed = time.monotonic() - start
        #print(elapsed*1000)
    #disparity = cv2.medianBlur(disparity, 5)
    repr_depths = cv2.reprojectImageTo3D(disparity, disparity_matrix, handleMissingValues=True)
    im_disp = np.clip(np.round(disparity), 0, 255).astype(np.uint8)
    im = np.clip(np.round(repr_depths), 0, 255).astype(np.uint8)
    #print(disparity.shape, im.shape)
    write_pointcloud(f"model_results/banet-{left_imgs.name}-{frame_id:06d}.ply", repr_depths, img_o)
    imageio.imwrite(f"model_results/banet-{left_imgs.name}-{frame_id:06d}-disp.png", im_disp)
    imageio.imwrite(f"model_results/banet-{left_imgs.name}-{frame_id:06d}.png", im)
    frame_id += 1
    # if frame_id > 5:
    #     break
