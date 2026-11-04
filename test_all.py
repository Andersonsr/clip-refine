#!/usr/bin/env python3
"""
Zero-shot evaluation (main/test.py) of every trained experiment in a results_dir, one after the other.

An experiment is a folder <results_dir>/<pattern>/experiment<id>/ holding a best_model*.pt and the config train.py
copied into it; it is evaluated with that config, so each run is tested with the model it was trained with.
Experiments without a model, or that already have a test_result.json (unless --overwrite), are skipped.

Usage (from the project root; extra arguments go to main/test.py):
    python test_all.py --results_dir /path/to/results --data_root /path/to/DATA
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from util import yaml_utils

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--data_root", required=True, help="folder given to download_datasets.sh --data")
    parser.add_argument("--overwrite", action="store_true", help="also evaluate experiments with a test_result.json")
    args, test_args = parser.parse_known_args()

    runs = []
    for out in sorted(glob.glob(os.path.join(args.results_dir, "*", "experiment*"))):
        match = re.fullmatch(r"experiment(\d+)", os.path.basename(out))
        if not match or not os.path.isdir(out):
            continue
        if not glob.glob(os.path.join(out, "best_model*.pt")):
            print(f"{out}: no best_model*.pt, skipped")
            continue
        if os.path.exists(os.path.join(out, "test_result.json")) and not args.overwrite:
            print(f"{out}: already has test_result.json, skipped")
            continue
        configs = glob.glob(os.path.join(out, "*.yaml")) + glob.glob(os.path.join(out, "*.yml"))
        if len(configs) != 1:
            print(f"{out}: expected one config, found {len(configs)}, skipped")
            continue
        with open(configs[0]) as f:
            pattern = yaml_utils.make_pattern(yaml_utils.Config(yaml.load(f, Loader=yaml.SafeLoader)))
        if pattern != os.path.basename(os.path.dirname(out)):
            print(f"{out}: its config gives the folder {pattern}, skipped")
            continue
        runs.append((out, int(match.group(1)), configs[0]))

    print(f"{len(runs)} experiments to evaluate:")
    for out, _, _ in runs:
        print(f"  {out}")
    failed = []
    with tempfile.TemporaryDirectory() as tmp:
        for out, experiment_id, config in runs:
            print(f"\n## {out}", flush=True)
            # test.py copies the config into the experiment folder: give it a copy, not the file already there
            config_copy = os.path.join(tmp, os.path.basename(config))
            shutil.copy(config, config_copy)
            command = [sys.executable, "main/test.py", "--config_path", config_copy, "--results_dir", args.results_dir,
                       "--data_root", args.data_root, "--experiment_id", str(experiment_id), *test_args]
            if subprocess.run(command, cwd=ROOT).returncode != 0:
                failed.append(out)
    if failed:
        sys.exit("failed:\n  " + "\n  ".join(failed))


if __name__ == "__main__":
    main()
