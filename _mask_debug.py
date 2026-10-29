import cv2
import numpy as np
from PIL import Image
img = cv2.imread(r"D:\\Disser\\Datasets\\temps\\TRAIN\\NEGATIVE\\REAL\\MY\\dataset-train-70\\5TJgISeAmYDg7IfJhHbMeXMViF2kz1SO6zOvZf9HCPhDpnP8JaJARuiAYrYzFWQTLSEt2YBrIzg3hTrNr0rw.jpg", cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]
print("Image size:", w, "x", h)
import onnxruntime as ort
sess = ort.InferenceSession(r"D:\\PSU\\slam-segm\\weights\\onnx\\rf-detr.onnx")
img_r = cv2.resize(img_rgb, (432, 432))
arr = img_r.astype(np.float32) / 255.0
tensor = arr.transpose(2,0,1)[None]
outs = sess.run(None, {"input": tensor})
logits = outs[1].squeeze()
masks_raw = outs[2].squeeze()
print("Raw masks shape:", masks_raw.shape)
probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
count = 0
for i in range(len(logits)):
    max_p = float(np.max(probs[i]))
    ci = int(np.argmax(probs[i]))
    if ci == 1 and max_p > 0.3:
        mr = masks_raw[i] > 0.0
        print("  det", i, "p=", round(max_p,3), "mask_sum:", int(mr.sum()), "shape:", mr.shape)
        count += 1
print("Total person detections:", count)