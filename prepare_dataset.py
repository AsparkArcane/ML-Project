import os
import shutil
import pandas as pd
import cv2
from pathlib import Path

def prepare_gtsrb(raw_gtsrb_path, target_gtsrb_path):
    """
    Organizes raw GTSRB data into data/GTSRB/Train/ and data/GTSRB/Test/
    by reading Test.csv to organize test images into numeric class folders (00000 -> 00042).
    """
    print("\n[1/3] Processing GTSRB Dataset...")
    raw_path = Path(raw_gtsrb_path)
    target_path = Path(target_gtsrb_path)
    
    train_target = target_path / "Train"
    test_target = target_path / "Test"
    
    os.makedirs(train_target, exist_ok=True)
    os.makedirs(test_target, exist_ok=True)

    # --- PROCESS TRAIN SET ---
    raw_train = raw_path / "Train" if (raw_path / "Train").exists() else raw_path / "train"
    if raw_train.exists():
        print("  └─ Organizing Training folders...")
        for class_dir in raw_train.iterdir():
            if class_dir.is_dir():
                try:
                    class_id = int(class_dir.name)
                    formatted_class_name = f"{class_id:05d}"
                    dest_dir = train_target / formatted_class_name
                    os.makedirs(dest_dir, exist_ok=True)
                    
                    for img_file in class_dir.glob("*.*"):
                        if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.ppm']:
                            shutil.copy(img_file, dest_dir / img_file.name)
                except ValueError:
                    continue
        print("  └─ Train set ready.")
    else:
        print("  [!] Warning: Could not find 'Train' directory inside raw GTSRB folder.")

    # --- PROCESS TEST SET USING Test.csv ---
    csv_file = None
    for p in [raw_path / "Test.csv", raw_path / "test.csv", raw_path / "GT-final_test.csv"]:
        if p.exists():
            csv_file = p
            break
            
    raw_test = raw_path / "Test" if (raw_path / "Test").exists() else raw_path / "test"

    if csv_file and raw_test.exists():
        print(f"  └─ Organizing Test images using CSV: {csv_file.name}...")
        df = pd.read_csv(csv_file)
        
        # Determine CSV column names (Kaggle vs Official SID benchmark format)
        path_col = 'Path' if 'Path' in df.columns else 'Filename'
        class_col = 'ClassId' if 'ClassId' in df.columns else 'ClassId'

        copied_count = 0
        for _, row in df.iterrows():
            img_rel_path = row[path_col]
            class_id = int(row[class_col])
            
            # Construct source file path
            src_file = raw_path / img_rel_path
            if not src_file.exists():
                src_file = raw_test / Path(img_rel_path).name

            if src_file.exists():
                formatted_class_name = f"{class_id:05d}"
                dest_dir = test_target / formatted_class_name
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy(src_file, dest_dir / src_file.name)
                copied_count += 1
                
        print(f"  └─ Successfully organized {copied_count} test images across 43 classes.")
    else:
        print("  [!] Test.csv or Test image folder not found. Moving raw test folder directly...")
        if raw_test.exists():
            for img_file in raw_test.glob("*.*"):
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.ppm']:
                    dest_dir = test_target / "00000"
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy(img_file, dest_dir / img_file.name)

def create_placeholders(data_root):
    """
    Creates empty class folder structures for BTSD and CTSD datasets.
    """
    print("\n[2/3] Setting up BTSD and CTSD Directory Trees...")
    for ds in ["BTSD", "CTSD"]:
        ds_path = Path(data_root) / ds
        for split in ["Train", "Test"]:
            for class_id in range(43):
                os.makedirs(ds_path / split / f"{class_id:05d}", exist_ok=True)
        print(f"  └─ Placeholders created for {ds} (00000 -> 00042).")

def verify_dataset(data_root):
    """
    Scans the dataset directory and logs image counts per dataset split.
    """
    print("\n[3/3] Verifying Final Dataset Structure...")
    data_path = Path(data_root)
    
    for ds_name in ["GTSRB", "BTSD", "CTSD"]:
        ds_dir = data_path / ds_name
        if not ds_dir.exists():
            continue
            
        train_count = len(list(ds_dir.glob("Train/*/*.*")))
        test_count = len(list(ds_dir.glob("Test/*/*.*")))
        
        print(f"  ├─ {ds_name}:")
        print(f"  │   ├─ Train Images : {train_count}")
        print(f"  │   └─ Test Images  : {test_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Traffic Sign Dataset Organization Tool")
    parser.add_argument("--raw_gtsrb", type=str, default=None, help="Path to your unzipped raw GTSRB folder")
    parser.add_argument("--output_dir", type=str, default="./data", help="Target project data directory")
    args = parser.parse_args()

    data_root = Path(args.output_dir)
    os.makedirs(data_root, exist_ok=True)

    if args.raw_gtsrb:
        prepare_gtsrb(args.raw_gtsrb, data_root / "GTSRB")
    else:
        print("\n[INFO] No raw GTSRB path passed. Assuming files are manually copied to ./data/GTSRB/")

    create_placeholders(data_root)
    verify_dataset(data_root)
    
    print("\n=========================================================")
    print("  Dataset Preparation Complete!")
    print("=========================================================\n")