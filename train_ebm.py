# train_ebm.py
import os, argparse, torch, torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from app.ebm_model import EnergyCNN

def langevin_sample(x, energy_fn, steps=60, step_size=10.0, noise=0.01):
    """
    Update images x by descending energy wrt inputs (NOT params).
    x.requires_grad_() so that gradients flow to x, not model weights.
    """
    x = x.clone().detach().requires_grad_(True)
    for _ in range(steps):
        e = energy_fn(x).sum()
        grad, = torch.autograd.grad(e, x, create_graph=False)
        x = x - step_size * grad
        x = x + noise * torch.randn_like(x)
        x = x.clamp(-1, 1).detach().requires_grad_(True)
    return x.detach()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load full CIFAR-10 training set
    ds_full = datasets.CIFAR10(root="data", train=True, download=True, transform=tf)

    # ✅ Use only a subset if requested
    if args.subset and args.subset > 0:
        ds = Subset(ds_full, range(min(args.subset, len(ds_full))))
        print(f"⚡ Using subset of {len(ds)} images from CIFAR-10")
    else:
        ds = ds_full
        print(f"📦 Using full CIFAR-10 dataset ({len(ds)} images)")

    dl = DataLoader(
        ds,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False
    )

    # Model + optimizer
    model = EnergyCNN().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    margin = 10.0  # Margin loss

    model.train()
    for epoch in range(1, args.epochs + 1):
        for (x, _) in dl:
            x = x.to(device)
            with torch.no_grad():
                x0 = torch.randn_like(x).clamp(-1, 1)  # init negatives
            x_neg = langevin_sample(
                x0, model, steps=30, step_size=1e-2, noise=0.01
            ).to(device)

            e_pos = model(x)     # lower is better
            e_neg = model(x_neg) # higher is better

            loss = torch.relu(margin + e_pos - e_neg).mean()
            loss = loss + 1e-4 * (e_pos ** 2).mean() + 1e-4 * (e_neg ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {epoch}: loss={loss.item():.4f}")

    # Save model
    os.makedirs("artifacts", exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, "artifacts/ebm_cifar10.pt")
    print("✅ Saved EBM model to artifacts/ebm_cifar10.pt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)  # 0 is safest on Windows
    ap.add_argument(
        "--subset",
        type=int,
        default=2000,
        help="Use only the first N images from CIFAR-10 (0 or omit for full dataset)."
    )
    args = ap.parse_args()
    main(args)
