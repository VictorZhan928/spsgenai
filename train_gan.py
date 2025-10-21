import os, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils as vutils
from app.gan_model import Generator, Discriminator

def get_loader(data_dir, batch_size):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

@torch.no_grad()
def save_grid(imgs, path, nrow=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vutils.save_image(imgs, path, nrow=nrow, normalize=True, value_range=(-1,1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--nz", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out", type=str, default="./artifacts/gan_mnist.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() 
                          else "mps" if torch.backends.mps.is_available() 
                          else "cpu")
    print("Using device:", device)

    loader = get_loader(args.data_dir, args.batch_size)
    G, D = Generator(args.nz).to(device), Discriminator().to(device)

    bce = nn.BCEWithLogitsLoss()
    optG = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_z = torch.randn(64, args.nz, device=device)

    for epoch in range(1, args.epochs+1):
        G.train(); D.train()
        g_loss_sum, d_loss_sum = 0, 0
        for real, _ in loader:
            real = real.to(device)
            N = real.size(0)
            ones, zeros = torch.ones(N, device=device), torch.zeros(N, device=device)

            # --- Train D ---
            z = torch.randn(N, args.nz, device=device)
            fake = G(z).detach()
            d_real = D(real)
            d_fake = D(fake)
            loss_d = bce(d_real, ones) + bce(d_fake, zeros)
            optD.zero_grad(); loss_d.backward(); optD.step()

            # --- Train G ---
            z = torch.randn(N, args.nz, device=device)
            fake = G(z)
            d_fake = D(fake)
            loss_g = bce(d_fake, ones)
            optG.zero_grad(); loss_g.backward(); optG.step()

            g_loss_sum += loss_g.item()
            d_loss_sum += loss_d.item()

        print(f"Epoch {epoch}: D={d_loss_sum/len(loader):.3f}, G={g_loss_sum/len(loader):.3f}")

        with torch.no_grad():
            samples = G(fixed_z).cpu()
            save_grid(samples, f"artifacts/epoch_{epoch:03d}.png")

    torch.save({"state_dict": G.state_dict(), "nz": args.nz}, args.out)
    print("✅ Saved generator to", args.out)

if __name__ == "__main__":
    main()
