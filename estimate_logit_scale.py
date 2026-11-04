#!/usr/bin/env python3
"""
Estimates the logit scale for training the model of a config on COCO or CC3M at the config's batch size.

1. Statistics of the training data under the model: cosine similarities of the matched image-text pairs (positives)
   and of all the mismatched pairs (negatives) of --num_samples random training samples, as median and sigma_MAD
   (1.4826 * median absolute deviation).
2. estimate_t2 (estimate_t2_monte_carlo.py) with these statistics on both sides: the model's pre-training logit scale
   T1 at the pre-training batch size --n1 is carried over to the config's batch size.

The result is added to estimated_logit_scale.json in the project root (one entry per backbone, dataset and batch size).

Usage (from the project root):
    python estimate_logit_scale.py --dataset coco --config config/01_post-pre-training/clip-refine.yaml --root /path/to/coco
    python estimate_logit_scale.py --dataset cc3m --config config/01_post-pre-training/clip-refine-cc3m.yaml --root /path/to/cc3m
"""

import argparse
import json
import math
import os

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from estimate_t2_monte_carlo import estimate_t2
from util import yaml_utils

DATASETS = {"coco": ("data/coco_caption.py", "COCOCaption"), "cc3m": ("data/cc3m.py", "CC3M")}
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "estimated_logit_scale.json")


def median_and_sigma_mad(x):
    median = x.median()
    return median.item(), 1.4826 * (x - median).abs().median().item()


@torch.no_grad()
def compute_statistics(model, dataset, num_samples=10000, batch_size=256, num_workers=8, seed=0, device="cuda"):
    """
    Similarities of the model's normalized image and text embeddings over num_samples random samples of dataset:
    positives are the matched pairs, negatives every image with the texts of the other samples.
    The similarity matrix takes about 45 bytes per pair at its peak: up to about 40,000 samples on an 80 GB GPU.

    Returns {"mu_p", "sigma_p", "mu_n", "sigma_n", "num_samples"} (medians and sigma_MAD).
    """
    torch.manual_seed(seed)  # the samples, and the caption COCOCaption picks for each image
    indices = torch.randperm(len(dataset))[:num_samples].tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, num_workers=num_workers)
    feats_i, feats_t = [], []
    for images, texts in tqdm(loader):
        feats_i.append(model.encode_image(images.to(device), normalized=True))
        feats_t.append(model.encode_text(texts.view(-1, texts.shape[-1]).to(device), normalized=True))
    sim = torch.cat(feats_i) @ torch.cat(feats_t).T
    negatives = ~torch.eye(len(sim), dtype=torch.bool, device=sim.device)
    mu_p, sigma_p = median_and_sigma_mad(sim.diagonal())
    mu_n, sigma_n = median_and_sigma_mad(sim[negatives])
    return {"mu_p": mu_p, "sigma_p": sigma_p, "mu_n": mu_n, "sigma_n": sigma_n, "num_samples": len(sim)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--config", required=True, help="training config: its model and batch size are used")
    parser.add_argument("--root", required=True, help="dataset folder (coco: train2017/ and annotations/; cc3m: shards)")
    parser.add_argument("--num_samples", type=int, default=10000, help="training samples the statistics come from")
    parser.add_argument("--n1", type=int, default=32768, help="batch size of the pre-training (32,768 for OpenAI CLIP)")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size for computing the embeddings")
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_config = config["models"]["model"]
    model = yaml_utils.load_model(model_config["func"], model_config["name"], model_config["args"]).to(device).eval()
    dataset = yaml_utils.load_module(*DATASETS[args.dataset])(model.tokenizer, model.img_preprocess, root=args.root)

    stats = compute_statistics(model, dataset, args.num_samples, args.batch_size, args.num_worker, args.seed, device)
    t1 = model.backbone.logit_scale.exp().item()  # the logit scale the backbone was pre-trained with
    n2 = config["batchsize"]
    result = estimate_t2(
        n1=args.n1,
        T1=t1,
        n2=n2,
        lhs_mu_p=stats["mu_p"],
        lhs_sigma_p=stats["sigma_p"],
        lhs_mu_n=stats["mu_n"],
        lhs_sigma_n=stats["sigma_n"],
        device=device,
    )

    entry = {
        "model": model_config["args"]["backbone_name"],
        "dataset": args.dataset,
        "config": args.config,
        "statistics": stats,
        "n1": args.n1,
        "T1": t1,
        "batch_size": n2,
        "logit_scale": result["T2_estimates"],
        # the same in log space, as models.model.args.init_logit_scale of a config takes it
        "init_logit_scale": {k: (math.log(v) if v else None) for k, v in result["T2_estimates"].items()},
    }
    entries = []
    if os.path.exists(OUTPUT):
        with open(OUTPUT) as f:
            entries = json.load(f)
    key = (entry["model"], entry["dataset"], entry["batch_size"])
    entries = [e for e in entries if (e["model"], e["dataset"], e["batch_size"]) != key] + [entry]
    with open(OUTPUT, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"saved to {OUTPUT}")


if __name__ == "__main__":
    main()
