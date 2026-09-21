from pathlib import Path
import json
import tqdm

base_imgs = Path("/home/quant/datasets/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/")
base_imgs2 = Path("/home/quant/datasets/cityscapes/leftImg8bit_trainextra/leftImg8bit/")
#base_label = Path("/home/quant/datasets/cityscapes/gtCoarse/gtCoarse/")
base_label = Path("/home/quant/datasets/cityscapes/gtFine_trainvaltest/gtFine/")
base_out = Path("/home/quant/datasets/cityscapes/coco_excluded")
base_out.mkdir(parents=True, exist_ok=True)

categories_raw = {
    "flat": ["road", "sidewalk", "parking rail", "track"], # TODO add comma
    "human": ["person", "rider"],
    "vehicle": ["car", "truck", "bus", "on rails", "motorcycle", "bicycle", "caravan", "trailer"],
    "construction": ["building", "wall", "fence", "guard rail", "bridge", "tunnel"],
    "object": ["pole", "pole group", "traffic sign", "traffic light"],
    "nature": ["vegetation", "terrain"],
    "sky": ["sky"],
    "void": ["ground", "dynamic", "static"],
}

exclude_categories = {'guard rail', 'bridge', 'dynamic', 'static'}

def convert_poly(poly):
    polygon = []
    min_x, min_y = poly[0]
    max_x, max_y = poly[0]
    for vertex in poly:
        polygon.extend(vertex)
        min_x = min(min_x, vertex[0])
        min_y = min(min_y, vertex[1])
        max_x = max(max_x, vertex[0])
        max_y = max(max_y, vertex[1])
    bbox = (
        min_x, min_y,
        max_x - min_x, max_y - min_y,
    )
    return bbox, polygon


# categories = {
#     {
#         "id": int, 
#         "name": str, 
#         "supercategory": str,
#     }
# }

def convert(split):

    nid = 0
    annotation_imgs = []
    annotation_anns = []
    annotation_categories = []
    label_map = dict()

    for supercategory, names in categories_raw.items():
        for name in names:
            cat_id = nid
            nid += 1
            annotation_categories.append({
                "id": cat_id,
                "name": name,
                "supercategory": supercategory,
            })
            label_map[name] = cat_id

    (base_out/split).mkdir(parents=True, exist_ok=True)
    for path in tqdm.tqdm(list((base_label/split).glob("**/*.json"))):
        relpath = path.relative_to(base_label/split)
        city = relpath.parent
        base_name, _, _ = relpath.name.rsplit("_", maxsplit=2)
        #print(relpath, city, base_name)
        rsplit = "train" if split == "train_extra_2" else split
        imgpath = base_imgs/rsplit/city/(base_name+"_leftImg8bit.png")
        if not imgpath.exists():
            imgpath = base_imgs2/"train_extra"/city/(base_name+"_leftImg8bit.png")

        assert imgpath.exists()
        #img = Image.open(imgpath)
        
        with open(path) as f:
            ann_data = json.load(f)
        
        image_id = nid
        nid += 1
        annotation_imgs.append({
            "id": image_id,
            "width": ann_data["imgWidth"],
            "height": ann_data["imgHeight"],
            "file_name": imgpath.name,
        })

        target = base_out/split/imgpath.name
        target.unlink(missing_ok=True)
        target.hardlink_to(imgpath)
        
        for obj in ann_data["objects"]:
            label = obj["label"]
            poly = obj["polygon"]
            
            if label not in label_map:
                continue
            if label in exclude_categories:
                continue
            
            bbox, polygon = convert_poly(poly)
            annotation_anns.append({
                "id": nid,
                "image_id": image_id,
                "category_id": label_map[label],
                "segmentation": [polygon],
                "area": 1.0, # TODO: calculate area
                "bbox": bbox,
                "iscrowd": 0,
            })
            nid += 1
        
    print("Writing json...")

    with open(base_out/split/"_annotations.coco.json", "w") as f:
        json.dump({
            "images": annotation_imgs,
            "annotations": annotation_anns,
            "categories": annotation_categories,
            "info": {"description": ""},
        }, f)

for split in ["val", "test", "train"]:
    convert(split)
#convert("train_extra_2")
