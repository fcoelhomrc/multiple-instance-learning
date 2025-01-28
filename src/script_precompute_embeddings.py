from mil_preprocessing import precompute_embeddings
import argparse
parser = argparse.ArgumentParser(description='Precompute embeddings')
parser.add_argument("--root_dir", type=str, required=True)
parser.add_argument("--patch_size", type=int, required=True)
parser.add_argument("--level", type=int, required=True)
parser.add_argument("--encoder_file", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=True)
args = parser.parse_args()

precompute_embeddings(args.root_dir, args.patch_size, args.level, args.encoder_file, args.batch_size)