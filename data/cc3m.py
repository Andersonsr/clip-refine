# CC3M read straight from WebDataset shards, ported from vlm-tuning/dataset/cc3m.py.
# The shards are indexed once and the index is cached in <root>/cc3m_<split>_index.npz (same file as vlm-tuning,
# so the two projects share it). Build it before training with: python data/cc3m.py --root /path/to/cc3m
import argparse
import io
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "webp")


def find_shards(root, split):
    # <root>/cc3m-<split>-NNNN.tar (hugging face mirror) or <root>/<split>/NNNNN.tar (img2dataset)
    shards = sorted(glob(os.path.join(root, f"*{split}*.tar")) + glob(os.path.join(root, split, "*.tar")))
    if not shards:
        raise FileNotFoundError(f"no {split} shards in {root}")
    return [os.path.relpath(path, root) for path in shards]


def index_shard(path):
    # (key, image offset, image size, caption offset, caption size) of every complete sample of a tar
    files = {}
    with tarfile.open(path) as tar:
        for member in tar:
            if not member.isfile():
                continue
            # webdataset convention: the key is the file name up to its first dot, the rest the field
            key, _, ext = os.path.basename(member.name).partition(".")
            files.setdefault(key, {})[ext.lower()] = (member.offset_data, member.size)

    samples = []
    for key, fields in files.items():
        image = next((fields[ext] for ext in IMAGE_EXTENSIONS if ext in fields), None)
        if image is not None and "txt" in fields:
            samples.append((key, *image, *fields["txt"]))
    return samples


def build_index(root, split, workers=8):
    shards = find_shards(root, split)
    path = os.path.join(root, f"cc3m_{split}_index.npz")
    if os.path.isfile(path):
        with np.load(path) as index:
            if index["shards"].tolist() == shards:
                return path
        print(f"{path} is outdated (the shards changed), rebuilding it")

    keys, shard_ids, entries = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = pool.map(index_shard, [os.path.join(root, s) for s in shards])
        for i, samples in enumerate(tqdm(results, total=len(shards), desc=f"indexing cc3m {split}")):
            for key, *offsets in samples:
                keys.append(key)
                shard_ids.append(i)
                entries.append(offsets)

    tmp = f"{path}.{os.getpid()}.tmp.npz"  # written whole, another process may be building it too
    np.savez(
        tmp,
        shards=np.array(shards),
        keys=np.array(keys, dtype="S"),
        shard_ids=np.array(shard_ids, dtype=np.int32),
        entries=np.array(entries, dtype=np.int64).reshape(-1, 4),
    )
    os.replace(tmp, path)
    return path


class CC3M(Dataset):

    def __init__(self, tokenizer, transform, test=False, root="./dataset/cc3m", split="train", workers=8):

        super().__init__()

        self.root = root
        # numpy arrays rather than python lists: no copy-on-read of 3M objects in every dataloader worker
        with np.load(build_index(root, split, workers)) as index:
            self.shards = index["shards"].tolist()
            self.shard_ids = index["shard_ids"]
            self.entries = index["entries"]

        self.files = {}  # open tars of this process

        self.transform = transform
        self.tokenizer = tokenizer

    def __getstate__(self):
        # dataloader workers open their own file handles
        state = self.__dict__.copy()
        state["files"] = {}
        return state

    def read(self, shard_id, offset, size):
        if shard_id not in self.files:
            self.files[shard_id] = open(os.path.join(self.root, self.shards[shard_id]), "rb")
        f = self.files[shard_id]
        f.seek(offset)
        return f.read(size)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        shard_id = int(self.shard_ids[idx])
        img_offset, img_size, text_offset, text_size = self.entries[idx].tolist()

        img = Image.open(io.BytesIO(self.read(shard_id, img_offset, img_size))).convert("RGB")
        text = self.read(shard_id, text_offset, text_size).decode("utf-8").strip()

        img_input = self.transform(img)
        text_input = self.tokenizer(text)

        return img_input, text_input


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index the CC3M shards (cached next to them) before training")
    parser.add_argument("--root", required=True)
    parser.add_argument("--splits", nargs="+", choices=["train", "validation"], default=["train"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    for split in args.splits:
        path = build_index(args.root, split, args.workers)
        with np.load(path) as index:
            print(f'{split}: {len(index["entries"])} samples in {len(index["shards"])} shards, index {path}')
