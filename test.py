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

DEVICE = "cuda"

export_model = True

if export_model:
    DEVICE = "cpu"

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

def zeros_like(x):
    return torch.zeros_like(x)

def fixed_batch_norm(x, a, bias, c, d, *args):
    print("batchnorm", a, bias, c, d)
    return torch.ops.aten.batch_norm.default(x, a, bias, c, d, *args)

def yeet_batchnorm(program):
    module = program.module()
    graph : torch.fx.Graph = torch.fx.Tracer().trace(module)

    for node in graph.nodes:
        if node.op == 'call_function' and node.target == torch.ops.aten.batch_norm.default:
            node.target = fixed_batch_norm

    graph.lint()

    return torch.fx.GraphModule(module, graph)
            

def do_export_model(banet, imgs):
    print("Exporting for shape", imgs[0].shape) #  torch.Size([1, 3, 416, 896])
    banet = banet.eval()

    batch = torch.export.Dim("batch")
    width = torch.export.Dim("width")
    height = torch.export.Dim("height")

    from copy import deepcopy
    banet.fnet_r = deepcopy(banet.fnet)
    #banet.stem_2_r = deepcopy(banet.stem_2)
    #banet.stem_4_r = deepcopy(banet.stem_4)

    # program = torch.onnx.export(banet, imgs, "vision_app/weights/onnx/banet_1.onnx", input_names=["left_image", "right_image"], dynamo=True, external_data=False, verify=False, dynamic_shapes=[
    #     {0: batch, 2: height, 3: width},
    #     {0: batch, 2: height, 3: width},
    # ], opset_version=18)

    program = torch.export.export(
        banet,
        imgs, 
    )

    #program = yeet_batchnorm(program)
    #program(*imgs)

    program = torch.onnx.export(
        program, 
        imgs, 
        "vision_app/weights/onnx/banet_1.onnx", 
        input_names=["left_image", "right_image"], 
        dynamo=True, 
        external_data=False, 
        verify=False, 
        opset_version=18,
        export_params=True,
        keep_initializers_as_inputs=False,
        do_constant_folding=True,
        optimize=False,
        dynamic_shapes=[
            {0: batch, 2: height, 3: width},
            {0: batch, 2: height, 3: width},
        ]
    )


    from torch.onnx import verification
    for val in verification.verify_onnx_program(program, args=imgs, compare_intermediates=False):
        if val.max_abs_diff > 3e-4:
            print(val.name, val.max_abs_diff, val.max_rel_diff)
            #print("\n======\n")
            #print(val)


#write_pointcloud("model_results/test.ply", np.ones((640, 480))*0.5)

import models
banet = models.load_banet(DEVICE)

from core.utils.utils import InputPadder

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


for path_left in tqdm.tqdm(sorted(list(left_imgs.glob("*.png")) + list(left_imgs.glob("*.jpg")))):
    if frame_id % 60 != 0:
        frame_id += 1
        continue
    
    with torch.no_grad():
        path_right = right_imgs / path_left.name
    #print(path_left, path_right)

        imgs = (convert(imageio.imread_v2(path_left)), convert(imageio.imread_v2(path_right)))
        img_o = imgs[0][1]
        imageio.imwrite(f"model_results/banet-{frame_id:06d}-orig.png", img_o)
        imgs = imgs[0][0], imgs[1][0]

        padder = InputPadder(imgs[0].shape, divis_by=32)
        imgs = padder.pad(*imgs)
        if export_model:
            do_export_model(banet, tuple(imgs))
            break
        start = time.monotonic()
        raw_disp = banet(*imgs)
        disparity = padder.unpad(raw_disp).cpu().numpy().squeeze()
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
