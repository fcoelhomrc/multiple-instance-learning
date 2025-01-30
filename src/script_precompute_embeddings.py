from mil_preprocessing import precompute_embeddings
import argparse
import torch

parser = argparse.ArgumentParser(description='Precompute embeddings')
parser.add_argument("--root_dir", type=str, required=True)
parser.add_argument("--patch_size", type=int, required=True)
parser.add_argument("--level", type=int, required=True)
parser.add_argument("--encoder", type=str, required=True)
parser.add_argument("--encoder_dir", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=True)

# Not required
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--auto_skip", action="store_true", default=False)

args = parser.parse_args()

if args.encoder == "resnet34":
    from torchvision import models
    encoder = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    encoder = torch.nn.Se#quential(*list(encoder.children())[:-1])
    encoder.eval()
else:
    raise NotImplementedError(args.encoder)


precompute_embeddings(
    root_dir=args.root_dir,
    patch_size=args.patch_size,
    level=args.level,
    encoder=encoder,
    encoder_dir=args.encoder_dir,
    batch_size=args.batch_size,
    device=args.device,
    auto_skip=args.auto_skip,
)
