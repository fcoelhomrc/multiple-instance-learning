from mil_preprocessing import create_splits
import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser(description='Precompute embeddings')
parser.add_argument("--root_dir", type=str, required=True)
parser.add_argument("--encoder", type=str, required=True)
parser.add_argument("--test_ratio", type=float, required=True)
parser.add_argument("--validation_ratio", type=float, required=True)
parser.add_argument("--stratify", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=42, required=False)

args = parser.parse_args()

splits = create_splits(
    root_dir=args.root_dir,
    test_ratio=args.test_ratio,
    validation_ratio=args.validation_ratio,
    stratified=args.stratify,
    seed=args.seed,
)

print("SPLITS")
print("=" * 100)
print(splits["split"].value_counts() / len(splits))
print("=" * 100)
