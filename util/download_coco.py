# Download COCO 2017 into the layout data/coco_caption.py expects:
#   <out>/train2017/, <out>/val2017/, <out>/annotations/captions_train2017.json
# Usage:
#   python util/download_coco.py --out ./dataset/coco
#   python util/download_coco.py --out ./dataset/coco --parts train annotations
import argparse
import os
import urllib.request
import zipfile

from tqdm import tqdm

URLS = {
    "train": "http://images.cocodataset.org/zips/train2017.zip",  # ~18GB
    "val": "http://images.cocodataset.org/zips/val2017.zip",  # ~1GB
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",  # ~240MB
}
EXTRACTED = {"train": "train2017", "val": "val2017", "annotations": "annotations"}


def fetch(url, path):
    if os.path.exists(path):
        print(f"{path} already exists, skipping download")
        return

    tmp_path = path + ".part"
    with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as f:
        total = int(response.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(path)) as bar:
            while chunk := response.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    os.replace(tmp_path, path)


def extract(zip_path, out_dir):
    with zipfile.ZipFile(zip_path) as zf:
        for member in tqdm(zf.infolist(), desc=f"extracting {os.path.basename(zip_path)}"):
            zf.extract(member, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download COCO 2017")
    parser.add_argument("--out", required=True, help="Destination folder")
    parser.add_argument("--parts", nargs="+", choices=list(URLS), default=list(URLS))
    parser.add_argument("--keep-zips", action="store_true", help="Keep the zips after extracting them")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    for part in args.parts:
        # an interrupted extraction leaves this folder half full: delete it to redo the part
        if os.path.isdir(os.path.join(args.out, EXTRACTED[part])):
            print(f"{EXTRACTED[part]} already extracted, skipping")
            continue
        zip_path = os.path.join(args.out, os.path.basename(URLS[part]))
        fetch(URLS[part], zip_path)
        extract(zip_path, args.out)
        if not args.keep_zips:
            os.remove(zip_path)

    print(f"COCO 2017 ({', '.join(args.parts)}) saved to {args.out}")
