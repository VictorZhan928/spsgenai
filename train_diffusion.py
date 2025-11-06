# train_diffusion.py
import os, argparse, torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils as vutils
from torch.utils.data import Subset, DataLoader
from app.diffusion_model import TinyUNet

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load full dataset
    ds_full = datasets.CIFAR10(root="data", train=True, download=True, transform=tf)

    # ✅ Use only subset if requested
    if args.subset and args.subset > 0:
        ds = Subset(ds_full, range(min(args.subset, len(ds_full))))
        print(f"⚡ Using subset of {len(ds)} images from CIFAR-10")
    else:
        ds = ds_full
        print(f"📦 Using full CIFAR-10 dataset ({len(ds)} images)")

    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=False)

    # Model + optimizer
    model = TinyUNet().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    mse = nn.MSELoss()

    # Diffusion hyperparameters
    T = args.T
    betas = torch.linspace(1e-4, 0.02, T).to(device)
    alphas = 1.0 - betas
    alpha_hat = torch.cumprod(alphas, dim=0)

    def q_sample(x0, t, noise):
        sqrt_alpha_hat = alpha_hat[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus = (1 - alpha_hat[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_alpha_hat * x0 + sqrt_one_minus * noise

    model.train()
    for epoch in range(1, args.epochs + 1):
        for x, _ in dl:
            x = x.to(device)
            t = torch.randint(0, T, (x.size(0),), device=device)
            noise = torch.randn_like(x)
            x_t = q_sample(x, t, noise)
            noise_pred = model(x_t, t)
            loss = mse(noise_pred, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {epoch}: loss={loss.item():.4f}")

    os.makedirs("artifacts", exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, "artifacts/diffusion_cifar10.pt")
    print("✅ Saved Diffusion model to artifacts/diffusion_cifar10.pt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--subset", type=int, default=2000,
                    help="Use only the first N images from CIFAR-10 (0 for full set).")
    ap.add_argument("--T", type=int, default=200,
                    help="Number of diffusion steps")
    args = ap.parse_args()
    main(args)
