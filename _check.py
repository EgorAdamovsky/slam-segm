import onnxruntime as ort; import numpy as np; import cv2
sess = ort.InferenceSession(r"D:\PSU\slam-segm\weights\onnx\rf-detr.onnx")
img = cv2.imread(r"D:\Disser\Datasets\temps\TRAIN\NEGATIVE\REAL\MY\dataset-train_10\lX9VV9oOKyM1GQ1otJmHn1yNI1G7zuKdPd9ntijzKPkFQzxRitg76Is8rpE1tvPnSnhRhXhguM1pCkSfeXcw.jpg", cv2.IMREAD_COLOR)
if img is None: print("Image not found"); exit()
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]
img_resized = cv2.resize(img_rgb, (432,432))
arr = img_resized.astype(np.float32)/255.0
tensor = arr.transpose(2,0,1)[None]
outs = sess.run(None, {"input": tensor})
dets = outs[0].squeeze(); labels = outs[1].squeeze()
print("Image size:", w, "x", h)
print("Checking all 200 detections...")
for i in range(len(dets)):
    ci = int(np.argmax(labels[i])); conf = labels[i][ci]
    if float(conf) > -1.0:  # raw logits, not sigmoid
        print(f"  det {i}: cls={ci} logit={float(conf):.4f} bbox={list(dets[i])}"
