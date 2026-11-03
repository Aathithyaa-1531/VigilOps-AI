"""
VigilOps AI - Dataset Preparation Pipeline for YOLO PPE Fine-Tuning
Converts Ultralytics Construction-PPE dataset + industrial machinery/surveillance datasets
into a unified 7-class YOLO dataset in datasets/ppe_finetune/.
Classes:
  0: hard_hat
  1: no_helmet
  2: reflective_jacket
  3: no_vest
  4: person
  5: forklift
  6: crane
"""

import os
import shutil
from collections import Counter
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = "/tmp/ppe_downloads/construction-ppe"
EXISTING_PPE_DIR = os.path.join(BASE_DIR, "datasets", "real_ppe_vision")
TARGET_DIR = os.path.join(BASE_DIR, "datasets", "ppe_finetune")

# Unified Class Mapping
CLASS_NAMES = {
    0: "hard_hat",
    1: "no_helmet",
    2: "reflective_jacket",
    3: "no_vest",
    4: "person",
    5: "forklift",
    6: "crane"
}

# Source Construction-PPE Mapping
# 0: helmet -> 0 (hard_hat)
# 7: no_helmet -> 1 (no_helmet)
# 2: vest -> 2 (reflective_jacket)
# 6: Person -> 4 (person)
SRC_CLASS_MAP = {
    0: 0,  # helmet -> hard_hat
    7: 1,  # no_helmet -> no_helmet
    2: 2,  # vest -> reflective_jacket
    6: 4   # Person -> person
}

def box_iou(b1, b2):
    # YOLO format: cx, cy, w, h
    x1_min, x1_max = b1[0] - b1[2]/2, b1[0] + b1[2]/2
    y1_min, y1_max = b1[1] - b1[3]/2, b1[1] + b1[3]/2
    x2_min, x2_max = b2[0] - b2[2]/2, b2[0] + b2[2]/2
    y2_min, y2_max = b2[1] - b2[3]/2, b2[1] + b2[3]/2
    
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    b1_area = b1[2] * b1[3]
    b2_area = b2[2] * b2[3]
    return inter_area / float(b1_area + b2_area - inter_area)

def process_construction_ppe():
    print("🚀 [Step 1] Preparing datasets/ppe_finetune from Construction-PPE...")
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(TARGET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(TARGET_DIR, "labels", split), exist_ok=True)
        
        src_img_dir = os.path.join(SOURCE_DIR, "images", split)
        src_lbl_dir = os.path.join(SOURCE_DIR, "labels", split)
        tgt_img_dir = os.path.join(TARGET_DIR, "images", split)
        tgt_lbl_dir = os.path.join(TARGET_DIR, "labels", split)
        
        class_counter = Counter()
        processed_files = 0
        
        for fname in os.listdir(src_img_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            base_name, _ = os.path.splitext(fname)
            lbl_name = base_name + ".txt"
            src_lbl_path = os.path.join(src_lbl_dir, lbl_name)
            
            if not os.path.exists(src_lbl_path):
                continue
            
            raw_boxes = []
            with open(src_lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        coords = [float(x) for x in parts[1:5]]
                        raw_boxes.append((cid, coords))
            
            target_annotations = []
            persons = []
            vests = []
            
            for cid, coords in raw_boxes:
                if cid in SRC_CLASS_MAP:
                    mapped_cid = SRC_CLASS_MAP[cid]
                    target_annotations.append((mapped_cid, coords))
                    class_counter[mapped_cid] += 1
                    if mapped_cid == 4: # person
                        persons.append(coords)
                    elif mapped_cid == 2: # vest
                        vests.append(coords)
            
            # Check for no_vest workers
            for p_coords in persons:
                # Approximate torso region of person:
                # center x = p_coords[0], center y = p_coords[1] + 0.05*p_coords[3], w = p_coords[2]*0.8, h = p_coords[3]*0.45
                p_torso = [p_coords[0], p_coords[1] - p_coords[3]*0.10, p_coords[2]*0.75, p_coords[3]*0.45]
                has_vest = False
                for v_coords in vests:
                    # Check center containment or IoU
                    if abs(v_coords[0] - p_coords[0]) < (p_coords[2] * 0.4) and abs(v_coords[1] - p_torso[1]) < (p_coords[3] * 0.3):
                        has_vest = True
                        break
                    if box_iou(p_torso, v_coords) > 0.20:
                        has_vest = True
                        break
                
                if not has_vest and p_coords[3] >= 0.12: # only for sufficiently sized visible workers
                    # Add localized no_vest box
                    target_annotations.append((3, p_torso))
                    class_counter[3] += 1
            
            if target_annotations:
                # Copy image
                src_img_path = os.path.join(src_img_dir, fname)
                tgt_img_path = os.path.join(tgt_img_dir, fname)
                shutil.copy2(src_img_path, tgt_img_path)
                
                # Write label
                tgt_lbl_path = os.path.join(tgt_lbl_dir, lbl_name)
                with open(tgt_lbl_path, "w") as f:
                    for cid, coords in target_annotations:
                        # format: cid cx cy w h
                        coords_str = " ".join(f"{max(0.0, min(1.0, c)):.6f}" for c in coords)
                        f.write(f"{cid} {coords_str}\n")
                
                processed_files += 1
        
        print(f"  [{split.upper()}] Processed {processed_files} images.")
        for cid, cnt in sorted(class_counter.items()):
            print(f"    Class {cid} ({CLASS_NAMES[cid]}): {cnt} instances")

def merge_machinery_and_cctv():
    print("\n🚀 [Step 2] Merging industrial machinery (forklift, crane) & CCTV frames...")
    # Merge existing curated forklift/crane annotations from real_ppe_vision
    exist_train_img = os.path.join(EXISTING_PPE_DIR, "images", "train")
    exist_train_lbl = os.path.join(EXISTING_PPE_DIR, "labels", "train")
    exist_val_img = os.path.join(EXISTING_PPE_DIR, "images", "val")
    exist_val_lbl = os.path.join(EXISTING_PPE_DIR, "labels", "val")
    
    tgt_train_img = os.path.join(TARGET_DIR, "images", "train")
    tgt_train_lbl = os.path.join(TARGET_DIR, "labels", "train")
    tgt_val_img = os.path.join(TARGET_DIR, "images", "val")
    tgt_val_lbl = os.path.join(TARGET_DIR, "labels", "val")
    tgt_test_img = os.path.join(TARGET_DIR, "images", "test")
    tgt_test_lbl = os.path.join(TARGET_DIR, "labels", "test")
    
    added_machinery = 0
    if os.path.exists(exist_train_img):
        for fname in os.listdir(exist_train_img):
            if fname.lower().endswith((".jpg", ".png")):
                base = os.path.splitext(fname)[0]
                lbl_path = os.path.join(exist_train_lbl, base + ".txt")
                if os.path.exists(lbl_path):
                    with open(lbl_path, "r") as f:
                        lines = [line.strip().split() for line in f if line.strip()]
                    has_machinery = any(parts[0] in ['5', '6'] for parts in lines if len(parts) >= 5)
                    if has_machinery:
                        new_name = f"plant_{fname}"
                        new_lbl = f"plant_{base}.txt"
                        shutil.copy2(os.path.join(exist_train_img, fname), os.path.join(tgt_train_img, new_name))
                        shutil.copy2(lbl_path, os.path.join(tgt_train_lbl, new_lbl))
                        added_machinery += 1
                        
    if os.path.exists(exist_val_img):
        for idx, fname in enumerate(sorted(os.listdir(exist_val_img))):
            if fname.lower().endswith((".jpg", ".png")):
                base = os.path.splitext(fname)[0]
                lbl_path = os.path.join(exist_val_lbl, base + ".txt")
                if os.path.exists(lbl_path):
                    with open(lbl_path, "r") as f:
                        lines = [line.strip().split() for line in f if line.strip()]
                    has_machinery = any(parts[0] in ['5', '6'] for parts in lines if len(parts) >= 5)
                    if has_machinery:
                        new_name = f"plant_{fname}"
                        new_lbl = f"plant_{base}.txt"
                        # Half to val, half to test
                        dest_i = tgt_val_img if (idx % 2 == 0) else tgt_test_img
                        dest_l = tgt_val_lbl if (idx % 2 == 0) else tgt_test_lbl
                        shutil.copy2(os.path.join(exist_val_img, fname), os.path.join(dest_i, new_name))
                        shutil.copy2(lbl_path, os.path.join(dest_l, new_lbl))
                        added_machinery += 1

    print(f"  Added {added_machinery} curated industrial machinery images (forklifts & cranes).")

def create_yaml():
    yaml_path = os.path.join(TARGET_DIR, "data.yaml")
    import yaml
    config = {
        "path": TARGET_DIR,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_NAMES
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"\n📄 Unified data configuration created: {yaml_path}")

if __name__ == "__main__":
    process_construction_ppe()
    merge_machinery_and_cctv()
    create_yaml()
    print("✅ Dataset preparation complete!")

