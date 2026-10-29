import os
import shutil
from pathlib import Path

def prepare_btsd(raw_btsd_path, target_btsd_path):
    """
    Parses and organizes raw Belgium Traffic Sign (BelgiumTSC/BTSD) data 
    from 'imagedb' and 'imagedb_test' directories into 5-digit formatted 
    subfolders inside data/BTSD/Train/ and data/BTSD/Test/.
    """
    print("\n=========================================================")
    print("  Belgium Traffic Sign Dataset (BTSD) Organizer")
    print("=========================================================\n")

    raw_path = Path(raw_btsd_path)
    target_path = Path(target_btsd_path)
    
    train_target = target_path / "Train"
    test_target = target_path / "Test"
    
    os.makedirs(train_target, exist_ok=True)
    os.makedirs(test_target, exist_ok=True)

    # 1. Search for Training Folder ('imagedb')
    raw_train = None
    for cand in [raw_path / "imagedb", raw_path / "BelgiumTSC" / "imagedb", raw_path]:
        if cand.exists() and any(cand.glob("00*")):
            raw_train = cand
            break

    # Process Training Set
    if raw_train and raw_train.exists():
        print(f"[1/3] Processing Training Folders from: {raw_train}")
        copied_train = 0
        for class_dir in sorted(raw_train.iterdir()):
            if class_dir.is_dir():
                try:
                    class_id = int(class_dir.name)
                    formatted_class = f"{class_id:05d}"
                    dest_dir = train_target / formatted_class
                    os.makedirs(dest_dir, exist_ok=True)

                    for img_file in class_dir.glob("*.*"):
                        if img_file.suffix.lower() in ['.ppm', '.png', '.jpg', '.jpeg']:
                            shutil.copy(img_file, dest_dir / img_file.name)
                            copied_train += 1
                except ValueError:
                    continue
        print(f"  └─ Copied {copied_train} Training images.")
    else:
        print("  [!] Error: Could not locate 'imagedb' training folder.")

    # 2. Search for Testing Folder ('imagedb_test')
    raw_test = None
    for cand in [raw_path / "imagedb_test", raw_path / "BelgiumTSC" / "imagedb_test", raw_path]:
        if cand.exists() and any(cand.glob("00*")):
            raw_test = cand
            break

    # Process Testing Set
    if raw_test and raw_test.exists():
        print(f"\n[2/3] Processing Testing Folders from: {raw_test}")
        copied_test = 0
        for class_dir in sorted(raw_test.iterdir()):
            if class_dir.is_dir():
                try:
                    class_id = int(class_dir.name)
                    formatted_class = f"{class_id:05d}"
                    dest_dir = test_target / formatted_class
                    os.makedirs(dest_dir, exist_ok=True)

                    for img_file in class_dir.glob("*.*"):
                        if img_file.suffix.lower() in ['.ppm', '.png', '.jpg', '.jpeg']:
                            shutil.copy(img_file, dest_dir / img_file.name)
                            copied_test += 1
                except ValueError:
                    continue
        print(f"  └─ Copied {copied_test} Testing images.")
    else:
        print("  [!] Error: Could not locate 'imagedb_test' testing folder.")

    # 3. Final Structure Verification
    print("\n[3/3] BTSD Final Structure Verification...")
    train_count = len(list(train_target.glob("*/*.*")))
    test_count = len(list(test_target.glob("*/*.*")))
    print(f"  ├─ BTSD Train Images: {train_count}")
    print(f"  └─ BTSD Test Images : {test_count}")
    print("\n=========================================================")
    print("  BTSD Dataset Preparation Complete!")
    print("=========================================================\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BTSD Dataset Organizer")
    parser.add_argument("--raw_btsd", type=str, required=True, help="Path to your unzipped BelgiumTSC folder")
    parser.add_argument("--output_dir", type=str, default="./data/BTSD", help="Target BTSD output folder")
    args = parser.parse_args()

    prepare_btsd(args.raw_btsd, args.output_dir)