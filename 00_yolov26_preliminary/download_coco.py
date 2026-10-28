"""
COCO Dataset Downloader for YOLOv26 Evaluation and Training.

Supports:
- 'mini' (coco8 / coco128): Lightweight sample subsets for immediate verification (~5-20 MB)
- 'val' (COCO val2017): Complete validation split of 5,000 images + annotations (~1 GB)
- 'full' (COCO train2017 + val2017): Complete training and validation sets (~20 GB)
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

def download_dataset(split: str = "val"):
    base_dir = Path(__file__).resolve().parent
    datasets_dir = base_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"============================================================")
    print(f" COCO Dataset Setup: split='{split}'")
    print(f" Target Directory: {datasets_dir}")
    print(f"============================================================")
    
    from ultralytics.utils.downloads import download
    from ultralytics.data.utils import check_det_dataset
    
    if split == "mini":
        print("[*] Downloading COCO8 mini benchmark (8 images, instant)...")
        dataset_yaml = "coco8.yaml"
        check_det_dataset(dataset_yaml)
        print("[+] COCO8 dataset ready!")
        return dataset_yaml

    elif split == "coco128":
        print("[*] Downloading COCO128 benchmark (128 images)...")
        dataset_yaml = "coco128.yaml"
        check_det_dataset(dataset_yaml)
        print("[+] COCO128 dataset ready!")
        return dataset_yaml

    elif split == "val":
        print("[*] Setting up COCO val2017 (5,000 images + YOLO labels)...")
        coco_dir = datasets_dir / "coco"
        images_val = coco_dir / "images" / "val2017"
        labels_val = coco_dir / "labels" / "val2017"
        
        # Check if already downloaded
        if images_val.exists() and len(list(images_val.glob("*.jpg"))) >= 4900 and labels_val.exists():
            print(f"[+] COCO val2017 already exists with {len(list(images_val.glob('*.jpg')))} images and labels.")
        else:
            if not (images_val.exists() and len(list(images_val.glob("*.jpg"))) >= 4900):
                # Download val2017 images (approx 1 GB)
                val_url = "http://images.cocodataset.org/zips/val2017.zip"
                print(f"[*] Downloading val2017 images from {val_url}...")
                download(val_url, dir=coco_dir / "images", unzip=True, delete=True)
            
            if not labels_val.exists():
                # Download YOLO labels into temp directory to avoid unzipping conflicts
                labels_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
                print(f"[*] Downloading YOLO format COCO labels from {labels_url}...")
                temp_labels = datasets_dir / "temp_labels"
                download(labels_url, dir=temp_labels, unzip=True, delete=True)
                if (temp_labels / "coco" / "labels").exists():
                    shutil.move(str(temp_labels / "coco" / "labels"), str(coco_dir / "labels"))
                shutil.rmtree(temp_labels, ignore_errors=True)
            
        # Create a dedicated coco_val.yaml configuration for validation
        coco_val_yaml = coco_dir / "coco_val.yaml"
        yaml_content = f"""# Ultralytics COCO val2017 dataset config
path: {coco_dir.resolve()}
train: images/val2017
val: images/val2017

names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
"""
        with open(coco_val_yaml, "w") as f:
            f.write(yaml_content)
            
        print(f"[+] COCO val2017 configuration saved to: {coco_val_yaml}")
        return str(coco_val_yaml)

    elif split == "full":
        print("[*] Downloading Full COCO 2017 dataset (train + val ~20 GB)...")
        dataset_yaml = "coco.yaml"
        check_det_dataset(dataset_yaml)
        print("[+] Full COCO dataset ready!")
        return dataset_yaml
    else:
        raise ValueError(f"Unknown split: {split}. Choose from ['mini', 'coco128', 'val', 'full']")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare COCO dataset for YOLOv26")
    parser.add_argument("--split", type=str, default="val", choices=["mini", "coco128", "val", "full"],
                        help="Dataset split to download: 'mini' (8 img), 'coco128' (128 img), 'val' (5000 img, ~1GB), or 'full' (~20GB)")
    args = parser.parse_args()
    
    cfg = download_dataset(args.split)
    print(f"\n[SUCCESS] Dataset ready. Use with YOLOv26: model.val(data='{cfg}')")
