#!/usr/bin/env python3
"""
Trains with each T2 estimate of an entry of estimated_logit_scale.json, one run after the other.

For the entry at --index, the training config the estimate was made with is copied once per estimate, with
models.model.args.init_logit_scale set to it, into <results_dir>/estimated_logit_scale_configs/. Each copy is trained
by main/train.py in the same --results_dir with its own experiment id: --first_experiment_id + the estimate's position
in the entry (T2_softmax_eq8, T2_log_softmax_normalized, T2_log_softmax_absolute, T2_quadratic_analytical). An estimate
without a value (no root found) is skipped, and its experiment id left unused.

Usage (from the project root; extra arguments go to main/train.py):
    python train_estimated_logit_scale.py --index 0 --results_dir /path/to/results --first_experiment_id 10
"""

import argparse
import json
import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=int, required=True, help="entry of estimated_logit_scale.json")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--first_experiment_id", type=int, default=0)
    args, train_args = parser.parse_known_args()

    with open(os.path.join(ROOT, "estimated_logit_scale.json")) as f:
        entry = json.load(f)[args.index]
    with open(os.path.join(ROOT, entry["config"])) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    config_dir = os.path.join(args.results_dir, "estimated_logit_scale_configs")
    os.makedirs(config_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(entry["config"]))[0]
    runs = []
    for position, (estimate, init_logit_scale) in enumerate(entry["init_logit_scale"].items()):
        if init_logit_scale is None:
            print(f"{estimate}: no estimate, skipped")
            continue
        config["models"]["model"]["args"]["init_logit_scale"] = init_logit_scale
        config_path = os.path.join(config_dir, f"{name}-{entry['dataset']}-bs{entry['batch_size']}-{estimate}.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, sort_keys=False)
        runs.append((args.first_experiment_id + position, estimate, config_path))

    print(f"entry {args.index}: {entry['model']} on {entry['dataset']}, batch size {entry['batch_size']}")
    for experiment_id, estimate, _ in runs:
        print(f"  experiment{experiment_id}: {estimate}, logit scale {entry['logit_scale'][estimate]:.4f}")
    failed = []
    for experiment_id, estimate, config_path in runs:
        print(f"\n## experiment{experiment_id}: {estimate}", flush=True)
        command = [sys.executable, "main/train.py", "--config_path", config_path, "--results_dir", args.results_dir,
                   "--experiment_id", str(experiment_id), *train_args]
        if subprocess.run(command, cwd=ROOT).returncode != 0:
            failed.append(f"experiment{experiment_id} ({estimate})")
    if failed:
        sys.exit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
