import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase, RFDETRSegPreview
import time
import os
import tqdm
from pathlib import Path
from torch.utils.flop_counter import FlopCounterMode
import torch

#model = RFDETRSegPreview(pretrain_weights="trained_models/att2/checkpoint_best_ema.pth", device="cuda")
model = RFDETRSegPreview(pretrain_weights="trained_models/att5/checkpoint_best_ema.pth", device="cpu")

# model.optimize_for_inference()

#classes = ['road', 'sidewalk', 'person', 'rider', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'caravan', 'trailer', 'building', 'wall', 'fence', 'guard rail', 'bridge', 'pole', 'traffic sign', 'traffic light', 'vegetation', 'terrain', 'sky', 'ground', 'dynamic', 'static', 'all', '???']

COLOR_PALETTE = sv.ColorPalette.from_hex([
    "77FA77",
    "A351FB",
    "FF4040",
    "FFA1A0",
    "FF7633",
    "FFB633",
    "D1D435",
    "4CFB12",
    "94CF1A",
    "40DE8A",
    "1B9640",
    "00D6C1",
    "2E9CAA",
    "00C4FF",
    "364797",
    "6675FF",
    "0019EF",
    "863AFF",
    "530087",
    "CD3AFF",
    "FF97CA",
    "FF39C9",
    "D13124",

    "A351FB",
    "FF4040",
    "FFA1A0",
    "FF7633",
    "FFB633",
    "D1D435",
    "4CFB12",
    "94CF1A",
    "40DE8A",
    "1B9640",
    "00D6C1",
    "2E9CAA",
    "00C4FF",
    "364797",
    "6675FF",
    "0019EF",
    "863AFF",
    "530087",
    "CD3AFF",
    "FF97CA",
    "FF39C9",
    "D13124",
])

os.makedirs("model_results", exist_ok=True)

base = Path("/home/quant/datasets/drivingstereo/rainy/left-image-half-size/")

print(model.class_names)

frame_id = 0
for frame_path in tqdm.tqdm(list(base.glob("*.jpg"))):
#for frame_id in tqdm.trange(10):

    image = Image.open(frame_path) #.resize((640, 480), Image.Resampling.LANCZOS)

    start = time.monotonic()
    detections = model.predict(image, threshold=0.4)
    # detections = count_flops(model.model.model, image)
    elapsed = time.monotonic() - start
    #print(elapsed*1000)


    labels = [
        f"{model.model.class_names[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator(color=COLOR_PALETTE).annotate(annotated_image, detections)
    annotated_image = sv.MaskAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.INDEX).annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator(color=COLOR_PALETTE, smart_position=True, text_scale=0.25, text_padding=5).annotate(annotated_image, detections, labels)

    annotated_image.save(f"model_results/rf-detr-{frame_id:06d}.png")
    frame_id += 1
#sv.plot_image(annotated_image)
