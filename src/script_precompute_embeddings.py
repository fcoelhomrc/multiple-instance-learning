from mil_preprocessing import precompute_embeddings
import argparse
parser = argparse.ArgumentParser(description='Precompute embeddings')
parser.add_argument("--root_dir", dtype=str, required=True)
parser.add_argument("--patch_size", dtype=int, required=True)
parser.add_argument("--level", dtype=int, required=True)
parser.add_argument("--encoder_file", dtype=str, required=True)
parser.add_argument("--batch_size", dtype=int, required=True)
args = parser.parse_args()

precompute_embeddings(args.root_dir, args.patch_size, args.level, args.encoder_file, args.batch_size)