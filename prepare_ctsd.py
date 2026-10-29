import os
import shutil
import random
import pandas as pd
import cv2
from pathlib import Path

def prepare_ctsd_kaggle(raw_ctsd_path, target_ctsd_path="./data/CTSD", test_ratio=0.25):
    """
    Parses Kaggle's 'dmitryyemelyanov/chinese-traffic-signs' dataset.
    Crops sign ROIs using bounding boxes from annotation CSV files,
    organizes crops into 5-digit category folders (00000 -> 00057),
    and creates a stratified Train/Test split.
    """
    print("\n=========================================================")
    print("  Chinese Traffic Sign Dataset (CTSD) Bounding-Box Crop & Organizer")
    print("=========================================================\n")

    raw_path = Path(raw_ctsd_path)
    target_path = Path(target_ctsd_path)
    train_target = target_path / "Train"
    test_target = target_path / "Test"

    # Ensure 58 class folders exist (00000 -> 00057)
    for class_id in range(58):
        formatted_class = f"{class_id:05d}"
        os.makedirs(train_target / formatted_class, exist_ok=True)
        os.makedirs(test_target / formatted_class, exist_ok=True)

    # 1. Locate Annotation CSV File
    csv_file = None
    for cand in [raw_path / "annotations.csv", raw_path / "train.csv", raw_path / "labels.csv"]:
        if cand.exists():
            csv_file = cand
            break

    if not csv_file:
        # Check subdirectories if dataset is nested
        csv_files = list(raw_path.glob("**/*.csv"))
        if csv_files:
            csv_file = csv_files[0]

    # 2. Locate Images Directory
    images_dir = None
    for cand in [raw_path / "images", raw_path / "images" / "images", raw_path]:
        if cand.exists() and len(list(cand.glob("*.png"))) > 0:
            images_dir = cand
            break

    if not images_dir:
        png_files = list(raw_path.glob("**/*.png"))
        if png_files:
            images_dir = png_files[0].parent

    print(f"[1/3] Using Annotations: {csv_file}")
    print(f"[1/3] Using Image Root  : {images_dir}")

    # 3. Read Annotations & Extract Cropped Sign ROIs
    df = pd.read_csv(csv_file)
    
    # Standardize Column Names
    col_map = {c.lower().strip(): c for c in df.columns}
    file_col = col_map.get('file_name', col_map.get('filename', col_map.get('file', 'file_name')))
    cat_col = col_map.get('category', col_map.get('classid', col_map.get('class', 'category')))
    
    x1_col = col_map.get('x1', 'x1')
    y1_col = col_map.get('y1', 'y1')
    x2_col = col_map.get('x2', 'x2')
    y2_col = col_map.get('y2', 'y2')

    cropped_count = 0
    class_crops = {i: [] for i in range(58)}

    print("\n[2/3] Cropping Sign ROIs from full scene images...")
    for idx, row in df.iterrows():
        filename = str(row[file_col]).strip()
        category = int(row[cat_col])
        
        img_path = images_dir / filename
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w, _ = img.shape
        
        # Bounding box coordinates
        x1 = max(0, int(row[x1_col]))
        y1 = max(0, int(row[y1_col]))
        x2 = min(w, int(row[x2_col]))
        y2 = min(h, int(row[y2_col]))

        # Extract cropped ROI
        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            continue

        crop_filename = f"crop_{idx}_{filename}"
        class_crops[category].append((crop, crop_filename))
        cropped_count += 1

    print(f"  └─ Extracted {cropped_count} cropped traffic sign ROIs across classes.")

    # 4. Stratified Split (75% Train / 25% Test)
    print("\n[3/3] Performing Stratified 75/25 Train-Test Split...")
    train_saved = 0
    test_saved = 0

    for cat_id, crops in class_crops.items():
        if not crops:
            continue
        
        random.seed(42)
        random.shuffle(crops)

        num_test = int(len(crops) * test_ratio)
        test_crops = crops[:num_test]
        train_crops = crops[num_test:]

        cat_folder = f"{cat_id:05d}"

        for crop_img, crop_name in train_crops:
            cv2.imwrite(str(train_target / cat_folder / crop_name), crop_img)
            train_saved += 1

        for crop_img, crop_name in test_crops:
            cv2.imwrite(str(test_target / cat_folder / crop_name), crop_img)
            test_saved += 1

    print("\n=========================================================")
    print(f"Final CTSD Structure Audit:")
    print(f"  ├─ CTSD Train Images : {train_saved}")
    print(f"  └─ CTSD Test Images  : {test_saved}")
    print("=========================================================\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CTSD Bounding-Box Cropper & Organizer")
    parser.add_argument("--raw_ctsd", type=str, required=True, help="Path to raw cached CTSD dataset")
    parser.add_argument("--output_dir", type=str, default="./data/CTSD", help="Output directory")
    args = parser.parse_args()

    prepare_ctsd_kaggle(args.raw_ctsd, args.output_dir)