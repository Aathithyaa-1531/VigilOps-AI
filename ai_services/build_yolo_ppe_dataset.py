"""
Generate Primary Edge Vision Dataset: PPE_DATASET_YOLOv8 + Safety Helmet and Reflective Jacket.
Format: YOLO format (.jpg + .txt normalized bbox).
Creates training and validation image sets with corresponding YOLO annotation files.
"""

import os
import random
import yaml
from PIL import Image, ImageDraw, ImageFont
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "edge_vision")

TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "images", "train")
VAL_IMG_DIR = os.path.join(DATASET_DIR, "images", "val")
TRAIN_LBL_DIR = os.path.join(DATASET_DIR, "labels", "train")
VAL_LBL_DIR = os.path.join(DATASET_DIR, "labels", "val")

for d in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LBL_DIR, VAL_LBL_DIR]:
    os.makedirs(d, exist_ok=True)

CLASS_NAMES = {
    0: "hard_hat",
    1: "no_helmet",
    2: "reflective_jacket",
    3: "no_vest",
    4: "person",
    5: "forklift",
    6: "crane"
}

def create_yaml_config():
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    config = {
        "path": DATASET_DIR,
        "train": "images/train",
        "val": "images/val",
        "names": CLASS_NAMES
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"📄 Created YOLOv8 data config: {yaml_path}")
    return yaml_path

def generate_scene(img_path: str, lbl_path: str, img_w=640, img_h=640):
    """
    Renders an industrial scene with workers, hard hats, reflective vests, and machinery,
    and writes precise normalized YOLO bboxes to the label file.
    """
    # Background industrial floor / warehouse
    bg_colors = [(28, 33, 44), (35, 42, 54), (45, 52, 64), (20, 24, 33)]
    bg_color = random.choice(bg_colors)
    img = Image.new("RGB", (img_w, img_h), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw industrial floor markings (yellow caution stripes, grid)
    for y_line in range(img_h // 2, img_h, 40):
        draw.line([(0, y_line), (img_w, y_line)], fill=(40, 50, 65), width=2)
    # Caution diagonal hazard stripe zone
    for x_s in range(0, img_w, 80):
        draw.line([(x_s, img_h - 60), (x_s + 40, img_h)], fill=(200, 160, 20), width=6)

    bboxes = [] # list of (class_id, x_center, y_center, width, height)

    num_workers = random.randint(1, 3)
    for w_i in range(num_workers):
        # Worker dimensions
        worker_h = random.randint(180, 280)
        worker_w = int(worker_h * 0.42)
        x_min = random.randint(50 + w_i * 180, min(img_w - worker_w - 50, 160 + w_i * 200))
        y_min = random.randint(img_h // 3, img_h - worker_h - 20)
        x_max = x_min + worker_w
        y_max = y_min + worker_h

        # Person bounding box
        p_xc = (x_min + x_max) / 2.0 / img_w
        p_yc = (y_min + y_max) / 2.0 / img_h
        p_w = worker_w / float(img_w)
        p_h = worker_h / float(img_h)
        bboxes.append((4, p_xc, p_yc, p_w, p_h))

        # Person silhouette / body
        draw.rectangle([x_min, y_min + int(worker_h * 0.3), x_max, y_max], fill=(45, 60, 80)) # legs/pants

        # Head / Helmet
        head_h = int(worker_h * 0.22)
        head_w = int(worker_w * 0.55)
        hx_min = x_min + (worker_w - head_w) // 2
        hy_min = y_min
        hx_max = hx_min + head_w
        hy_max = hy_min + head_h

        has_helmet = random.choice([True, False, True]) # 66% compliant
        if has_helmet:
            # Yellow or White ANSI Hard Hat
            helmet_color = random.choice([(255, 215, 0), (255, 255, 255), (255, 140, 0)])
            draw.chord([hx_min - 2, hy_min, hx_max + 2, hy_max + 4], 180, 360, fill=helmet_color, outline=(20, 20, 20))
            # Helmet bbox
            h_xc = (hx_min + hx_max) / 2.0 / img_w
            h_yc = (hy_min + hy_max) / 2.0 / img_h
            h_w = (head_w + 4) / float(img_w)
            h_h = (head_h + 4) / float(img_h)
            bboxes.append((0, h_xc, h_yc, h_w, h_h)) # 0 = hard_hat
        else:
            # Unhelmeted head
            draw.ellipse([hx_min, hy_min, hx_max, hy_max], fill=(180, 140, 110))
            h_xc = (hx_min + hx_max) / 2.0 / img_w
            h_yc = (hy_min + hy_max) / 2.0 / img_h
            h_w = head_w / float(img_w)
            h_h = head_h / float(img_h)
            bboxes.append((1, h_xc, h_yc, h_w, h_h)) # 1 = no_helmet

        # Torso / Safety Vest
        torso_y_min = hy_max
        torso_y_max = torso_y_min + int(worker_h * 0.40)
        has_vest = random.choice([True, False, True]) # 66% compliant
        if has_vest:
            # Fluorescent high-vis green/orange vest with reflective silver stripes
            vest_color = random.choice([(57, 255, 20), (255, 110, 0), (220, 255, 0)])
            draw.rectangle([x_min, torso_y_min, x_max, torso_y_max], fill=vest_color)
            # Silver retroreflective bands
            mid_y = (torso_y_min + torso_y_max) // 2
            draw.line([(x_min, mid_y), (x_max, mid_y)], fill=(230, 230, 240), width=6)
            draw.line([(x_min + worker_w // 3, torso_y_min), (x_min + worker_w // 3, torso_y_max)], fill=(230, 230, 240), width=4)
            draw.line([(x_max - worker_w // 3, torso_y_min), (x_max - worker_w // 3, torso_y_max)], fill=(230, 230, 240), width=4)

            v_xc = (x_min + x_max) / 2.0 / img_w
            v_yc = (torso_y_min + torso_y_max) / 2.0 / img_h
            v_w = worker_w / float(img_w)
            v_h = (torso_y_max - torso_y_min) / float(img_h)
            bboxes.append((2, v_xc, v_yc, v_w, v_h)) # 2 = reflective_jacket
        else:
            # Dark non-reflective shirt
            draw.rectangle([x_min, torso_y_min, x_max, torso_y_max], fill=(40, 45, 55))
            v_xc = (x_min + x_max) / 2.0 / img_w
            v_yc = (torso_y_min + torso_y_max) / 2.0 / img_h
            v_w = worker_w / float(img_w)
            v_h = (torso_y_max - torso_y_min) / float(img_h)
            bboxes.append((3, v_xc, v_yc, v_w, v_h)) # 3 = no_vest

    # Occasionally add Forklift or Crane machinery
    if random.random() > 0.4:
        mach_type = random.choice([5, 6])
        if mach_type == 5: # Forklift
            fx_min, fy_min = random.randint(img_w // 2, img_w - 180), random.randint(img_h // 2, img_h - 140)
            fx_max, fy_max = fx_min + 160, fy_min + 120
            draw.rectangle([fx_min, fy_min, fx_max, fy_max], fill=(220, 160, 20), outline=(20, 20, 20), width=2)
            draw.rectangle([fx_min - 25, fy_max - 20, fx_min, fy_max - 5], fill=(80, 80, 80)) # fork
            f_xc = (fx_min + fx_max) / 2.0 / img_w
            f_yc = (fy_min + fy_max) / 2.0 / img_h
            f_w = 160 / float(img_w)
            f_h = 120 / float(img_h)
            bboxes.append((5, f_xc, f_yc, f_w, f_h))
        else: # Crane Hoist
            cx_min, cy_min = random.randint(50, img_w - 200), 20
            cx_max, cy_max = cx_min + 180, cy_min + 100
            draw.rectangle([cx_min, cy_min, cx_max, cy_max], fill=(180, 30, 30), outline=(20, 20, 20), width=3)
            draw.line([(cx_min + 90, cy_max), (cx_min + 90, cy_max + 60)], fill=(120, 120, 120), width=4) # cable
            c_xc = (cx_min + cx_max) / 2.0 / img_w
            c_yc = (cy_min + cy_max) / 2.0 / img_h
            c_w = 180 / float(img_w)
            c_h = 100 / float(img_h)
            bboxes.append((6, c_xc, c_yc, c_w, c_h))

    # Save JPEG
    img.save(img_path, quality=92)

    # Save YOLO normalized bbox TXT
    with open(lbl_path, "w") as f:
        for cls_id, xc, yc, w, h in bboxes:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

def generate_ppe_dataset(train_count=60, val_count=15):
    print("🎨 Synthesizing Primary Edge Vision Dataset (PPE_DATASET_YOLOv8 + Safety Helmet & Reflective Jacket)...")
    yaml_path = create_yaml_config()

    for i in range(1, train_count + 1):
        img_p = os.path.join(TRAIN_IMG_DIR, f"ppe_train_{i:04d}.jpg")
        lbl_p = os.path.join(TRAIN_LBL_DIR, f"ppe_train_{i:04d}.txt")
        generate_scene(img_p, lbl_p)

    for i in range(1, val_count + 1):
        img_p = os.path.join(VAL_IMG_DIR, f"ppe_val_{i:04d}.jpg")
        lbl_p = os.path.join(VAL_LBL_DIR, f"ppe_val_{i:04d}.txt")
        generate_scene(img_p, lbl_p)

    print(f"✅ Generated {train_count} Train and {val_count} Val YOLOv8 image & label pairs.")
    print(f"   Image format: .jpg | Label format: .txt normalized bboxes (classes: {list(CLASS_NAMES.values())})")
    return yaml_path

if __name__ == "__main__":
    generate_ppe_dataset()
