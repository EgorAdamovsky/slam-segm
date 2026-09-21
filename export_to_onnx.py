import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase, RFDETRSegPreview
import time
import os
import tqdm
from pathlib import Path

model = RFDETRSegPreview(pretrain_weights="trained_models/att5/checkpoint_best_ema.pth", device="cpu")
model.export(output_dir="weights/onnx")
