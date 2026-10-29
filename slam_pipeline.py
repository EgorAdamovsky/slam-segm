#!/usr/bin/env python3
"""SLAM pipeline: depth + segmentation from stereo pair or single image."""
import argparse, json, math, sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
import supervision as sv
try:
    import onnxruntime as ort
except ImportError:
    print("pip install onnxruntime supervision")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
BANET_MODEL = SCRIPT_DIR / "weights" / "onnx" / "banet_1.onnx"
RFDETR_MODEL = SCRIPT_DIR / "weights" / "onnx" / "rf-detr.onnx"
CAMERA_MATRIX = np.array([[1,0,0,-713.58],[0,1,0,-570.93],[0,0,0,1164.62],[1,0,1.9626,0]], dtype=np.float32)
SEG_SIZE = 432; CONF_TH = 0.3; IOU_TH = 0.5; MAX_DET = 30

def sigmoid(x):
    if isinstance(x, np.ndarray): return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))

def compute_iou(b1, b2):
    x1=max(b1[0],b2[0]); y1=max(b1[1],b2[1])
    x2=min(b1[0]+b1[2],b2[0]+b2[2]); y2=min(b1[1]+b1[3],b2[1]+b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1); a1=b1[2]*b1[3]; a2=b2[2]*b2[3]; u=a1+a2-inter
    return inter/u if u>0 else 0

def nms_filter(dets, iou_t=IOU_TH, max_k=MAX_DET):
    if not dets: return []
    sd=sorted(dets, key=lambda d:d["confidence"], reverse=True)
    keep=[]
    for d in sd:
        if len(keep)>=max_k: break
        ok=True
        for k in keep:
            bk=(k["bbox"]["cx"]-k["bbox"]["w"]/2,k["bbox"]["cy"]-k["bbox"]["h"]/2,k["bbox"]["w"],k["bbox"]["h"])
            dd=(d["bbox"]["cx"]-d["bbox"]["w"]/2,d["bbox"]["cy"]-d["bbox"]["h"]/2,d["bbox"]["w"],d["bbox"]["h"])
            if compute_iou(bk,dd)>iou_t: ok=False; break
        if ok: keep.append(d)
    return keep

def vis_seg(image, detections):
    arr = image.copy()
    if not detections: return arr
    overlay = arr.copy()
    for i, d in enumerate(detections):
        mask_uint8 = (d["mask"] * 255).astype(np.uint8)
        mask_resized = cv2.resize(mask_uint8, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_NEAREST)
        color = np.array([(i * 40) % 255, ((i * 40 + 85) % 255), ((i * 40 + 170) % 255)], dtype=np.uint8)
        mask_bool = mask_resized > 0
        for j in range(3):
            overlay[mask_bool, j] = (overlay[mask_bool, j].astype(np.int16) * 0.7 + color[j] * 0.3).astype(np.uint8)
    arr = overlay
    for i, d in enumerate(detections):
        b = d["bbox"]
        x1, y1 = int(b["cx"]-b["w"]/2), int(b["cy"]-b["h"]/2)
        x2, y2 = int(b["cx"]+b["w"]/2), int(b["cy"]+b["h"]/2)
        cv2.rectangle(arr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = d["class"] + " " + str(round(d["confidence"], 2))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(arr, (x1, y1-th-5), (x1+tw, y1), (0, 0, 0), -1)
        cv2.putText(arr, label, (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return arr

class DepthModel:
    MODEL_H = 736; MODEL_W = 1280
    def __init__(self, mp, providers=None):
        if providers is None: providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(mp), providers=providers)
        print("[BANet] Model loaded: " + str(mp.name))
    def __call__(self, left, right):
        oh, ow = left.shape[:2]
        lr = cv2.resize(left, (self.MODEL_W, self.MODEL_H), interpolation=cv2.INTER_CUBIC)
        rr = cv2.resize(right, (self.MODEL_W, self.MODEL_H), interpolation=cv2.INTER_CUBIC)
        def ti(img): return img.transpose(2,0,1)[None].astype(np.float32)
        inp = [i.name for i in self.session.get_inputs()]
        feeds = {inp[0]: ti(lr), inp[1]: ti(rr)}
        outs = self.session.run(None, feeds)
        return outs[0].squeeze()[:oh, :ow].astype(np.float32)

class SegmentationModel:
    def __init__(self, mp, providers=None):
        if providers is None: providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(mp), providers=providers)
        self.iname = self.session.get_inputs()[0].name
        print("[RF-DETR] Model loaded: " + str(mp.name))
    def __call__(self, image):
        oh, ow = image.shape[:2]
        img_r = cv2.resize(image, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_CUBIC)
        arr = img_r.astype(np.float32) / 255.0
        tensor = arr.transpose(2,0,1)[None]
        outs = self.session.run(None, {self.iname: tensor})
        dets = outs[0].squeeze(); logits = outs[1].squeeze(); masks = outs[2].squeeze()
        probs = sigmoid(logits)
        all_dets = []
        for i in range(len(dets)):
            max_p = float(np.max(probs[i])); ci = int(np.argmax(probs[i]))
            if ci == 1 and max_p > CONF_TH: semantic = "Person"
            elif ci in (3, 6, 8) and max_p > CONF_TH: semantic = "Car"
            else: continue
            cx, cy, w, h = dets[i]
            mask_raw = masks[i] > 0.0
            all_dets.append({"class": semantic, "confidence": max_p, "bbox": {"cx": float(cx*ow), "cy": float(cy*oh), "w": float(w*ow), "h": float(h*oh)}, "mask": mask_raw})
        return nms_filter(all_dets)

class SLAMPipeline:
    def __init__(self, banet_path=None, rfdetr_path=None, providers=None):
        banet_path = banet_path or BANET_MODEL
        rfdetr_path = rfdetr_path or RFDETR_MODEL
        if not banet_path.exists(): raise FileNotFoundError("BANet not found")
        if not rfdetr_path.exists(): raise FileNotFoundError("RF-DETR not found")
        self.depth_model = DepthModel(banet_path, providers)
        self.seg_model = SegmentationModel(rfdetr_path, providers)
        self.camera_matrix = CAMERA_MATRIX
    def process_frame(self, left_path, right_path):
        left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left is None or right is None: raise FileNotFoundError("Cannot load: " + str(left_path))
        left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB); right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
        oh, ow = left.shape[:2]
        disparity = self.depth_model(left, right)
        detections = self.seg_model(left)
        points_3d = disparity_to_3d(disparity, oh, ow, self.camera_matrix)
        class_masks, class_confs = {}, {}
        for det in detections:
            cls = det["class"]
            class_masks[cls] = np.maximum(class_masks.get(cls, np.zeros((oh, ow), dtype=bool)), det["mask"])
            class_confs[cls] = max(class_confs.get(cls, 0.0), det["confidence"])
        objects = extract_objects(points_3d, class_masks, class_confs)
        return {"left_img": left, "right_img": right, "disparity": disparity, "detections": detections, "objects": objects, "points_3d": points_3d}
    def process_single_image(self, img_path, out_dir):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = self.seg_model(img)
        return img, detections
    def save_results(self, frame_id, results, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = str(frame_id).zfill(6)
        left = results["left_img"]; disparity = results["disparity"]
        seg_vis = vis_seg(left, results["detections"])
        cv2.imwrite(str(out_dir / (prefix + "-seg.png")), cv2.cvtColor(seg_vis, cv2.COLOR_RGB2BGR))
        depth_vis = np.clip(disparity, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / (prefix + "-depth.png")), depth_vis)
        obj_list = []
        for v in results["objects"]["vehicles"]: obj_list.append({"type": "vehicle", "position": v["position"], "confidence": v["confidence"]})
        for p in results["objects"]["people"]: obj_list.append({"type": "person", "position": p["position"], "confidence": p["confidence"]})
        with open(out_dir / (prefix + "-objects.json"), "w") as fj: json.dump(obj_list, fj, indent=2)
        self._save_ply(out_dir / (prefix + "-points.ply"), results["points_3d"], left)
    @staticmethod
    def _save_ply(path, points_3d, colors):
        valid = (points_3d[:,:,0] != 0) & (points_3d[:,:,1] != 0) & (points_3d[:,:,2] != 0)
        pts = points_3d[valid]; cols = colors[valid]
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\nelement vertex " + str(len(pts)) + "\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for (x, y, z), (r, g, b) in zip(pts, cols):
                f.write(str(round(x,2)) + " " + str(round(y,2)) + " " + str(round(z,2)) + " " + str(int(r)) + " " + str(int(g)) + " " + str(int(b)) + "\n")

def disparity_to_3d(disparity, oh, ow, cm):
    pts = np.zeros((oh, ow, 3), dtype=np.float32)
    ys, xs = np.where(disparity[:oh, :ow] > 0)
    for y, x in zip(ys, xs):
        d = float(disparity[y, x]); pc = np.array([float(x), float(y), d], dtype=np.float32)
        pw = cm @ np.append(pc, 1.0); pts[y, x] = pw[:3]
    return pts

def extract_objects(p3d, masks, confs):
    vehicles, people = [], []
    for cn, mask in masks.items():
        if cn in ("Road", "Other"): continue
        ys, xs = np.where(mask)
        if len(xs) == 0: continue
        op = p3d[ys, xs]; info = {"class": cn, "confidence": confs.get(cn, 0.0), "point_count": int(len(xs))}
        if cn == "Car":
            dists = np.linalg.norm(op, axis=1); ci = int(np.argmin(dists))
            info["position"] = op[ci].tolist(); info["center"] = op.mean(axis=0).tolist()
            info["extents"] = {"min": op.min(axis=0).tolist(), "max": op.max(axis=0).tolist()}
            vehicles.append(info)
        elif cn == "Person":
            dists = np.linalg.norm(op, axis=1); ci = int(np.argmin(dists))
            info["position"] = op.mean(axis=0).tolist(); info["closest_to_camera"] = op[ci].tolist()
            people.append(info)
    return {"vehicles": vehicles, "people": people}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--out", type=str, default="model_results")
    parser.add_argument("--providers", type=str, default="CPUExecutionProvider")
    return parser.parse_args()

def print_summary(results):
    obj = results["objects"]
    print("  Vehicles: " + str(len(obj["vehicles"])) + ", People: " + str(len(obj["people"])))

def main():
    args = parse_args()
    providers = [p.strip() for p in args.providers.split(",")]
    if "CUDAExecutionProvider" in providers:
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            print("CUDA not available, using CPU"); providers = ["CPUExecutionProvider"]
    print("Providers: " + str(providers))
    pipeline = SLAMPipeline(providers=providers)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    if args.image:
        img, detections = pipeline.process_single_image(args.image, out_dir)
        print("[Segmentation] Found " + str(len(detections)) + " objects")
        for det in detections: print("  " + det["class"] + " conf=" + str(round(det["confidence"],3)))
        vis = vis_seg(img, detections)
        cv2.imwrite(str(out_dir / "seg_output.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print("Saved: " + str(out_dir / "seg_output.png"))
    else:
        print("Use --image <path>")
        sys.exit(1)

if __name__ == "__main__":
    main()
