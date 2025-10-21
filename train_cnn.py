# train_cnn.py
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from app.cnn_model import SimpleCNN


def get_loaders(data_dir: str,
                batch_size: int = 128,
                pin_mem: bool = False,
                num_workers: int = 2):
    """Build CIFAR-10 loaders with 64x64 resizing and normalization."""
    tf_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    tf_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_ds = datasets.CIFAR10(root=data_dir, train=True,
                                download=True, transform=tf_train)
    test_ds = datasets.CIFAR10(root=data_dir, train=False,
                               download=True, transform=tf_test)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_mem
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem
    )
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total if total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out", type=str, default="./artifacts/cnn_cifar10.pt")
    args = ap.parse_args()

    # Ensure output directory exists
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Pick device: MPS (Apple GPU) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # Dataloader tuning per device
    pin_mem = (device.type == "cuda")
    num_workers = 0 if device.type == "mps" else 2  # macOS MPS is happiest with 0

    # Model / data / optimizer
    model = SimpleCNN(num_classes=10).to(device)
    train_loader, test_loader = get_loaders(
        args.data_dir, args.batch_size, pin_mem=pin_mem, num_workers=num_workers
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item()

        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | "
              f"loss={running/len(train_loader):.4f} | "
              f"test_acc={test_acc:.3f}")

    # Save weights
    torch.save({"state_dict": model.state_dict()}, args.out)
    print(f"Saved weights to {args.out}")


if __name__ == "__main__":
    main()
