import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import load_data
from model import Autoencoder


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.as_tensor(x, dtype=torch.float32)


def _build_loaders(train_tensor, val_tensor, batch_size):
    train_ds = TensorDataset(train_tensor)
    val_ds = TensorDataset(val_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def _train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running = 0.0
    count = 0
    for (x,) in loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon = model(x)
        loss = criterion(recon, x)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
        count += x.size(0)
    return running / max(count, 1)


def _evaluate(model, loader, criterion, device):
    model.eval()
    running = 0.0
    count = 0
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            recon = model(x)
            loss = criterion(recon, x)
            running += loss.item() * x.size(0)
            count += x.size(0)
    return running / max(count, 1)


def _compute_threshold(model, loader, device, quantile):
    model.eval()
    errors = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            recon = model(x)
            batch_err = (recon - x).pow(2).mean(dim=1)
            errors.append(batch_err.cpu())
    all_errors = torch.cat(errors, dim=0)
    return torch.quantile(all_errors, quantile).item()


def load_model(weights_path, input_dim, latent_dim, device):
    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_threshold(threshold_path):
    payload = torch.load(threshold_path, map_location="cpu")
    return float(payload["threshold"])


def _collect_errors(model, loader, device):
    model.eval()
    errors = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            recon = model(x)
            batch_err = (recon - x).pow(2).mean(dim=1)
            errors.append(batch_err.cpu())
    if not errors:
        return torch.empty(0)
    return torch.cat(errors, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="telemetry.parquet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="artifacts")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_data, val_data, input_dim, meta = load_data(
        path=args.data,
        seed=args.seed
    )
    train_tensor = _to_tensor(train_data)
    val_tensor = _to_tensor(val_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Autoencoder(input_dim=input_dim, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    train_loader, val_loader = _build_loaders(train_tensor, val_tensor, args.batch_size)

    best_val = None
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = _evaluate(model, val_loader, criterion, device)
        if best_val is None or val_loss < best_val:
            best_val = val_loss
        print(f"epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    val_errors = _collect_errors(model, val_loader, device)
    if val_errors.numel() == 0:
        raise RuntimeError("No validation samples available to compute threshold.")
    threshold = torch.quantile(val_errors, args.threshold_quantile).item()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    weights_path = outdir / "autoencoder.pt"
    threshold_path = outdir / "threshold.pt"
    metadata_path = outdir / "metadata.json"

    torch.save(model.state_dict(), weights_path)
    torch.save({"threshold": threshold}, threshold_path)

    metadata = {
        "feature_cols": meta["feature_cols"],
        "scaler_mean": meta["mean"],
        "scaler_std": meta["std"],
        "window_length": meta["window_length"],
        "threshold": threshold,
        "model_checkpoint": str(weights_path),
        "data_path": meta["data_path"],
        "seed": args.seed,
        "latent_dim": args.latent_dim,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    reloaded_model = load_model(weights_path, input_dim, args.latent_dim, device)
    reloaded_threshold = load_threshold(threshold_path)
    _ = _collect_errors(reloaded_model, val_loader, device)

    anomaly_rate = (val_errors > reloaded_threshold).float().mean().item() * 100.0
    print(f"anomaly_rate_val={anomaly_rate:.2f}% (threshold={reloaded_threshold:.6f})")

    val_fault = meta.get("val_fault")
    if val_fault is not None:
        fault_tensor = torch.as_tensor(val_fault, dtype=torch.int64)
        fault0 = val_errors[fault_tensor == 0]
        fault1 = val_errors[fault_tensor == 1]
        if fault0.numel() > 0:
            print(f"val_fault0_err_mean={fault0.mean().item():.6f} std={fault0.std(unbiased=False).item():.6f}")
        if fault1.numel() > 0:
            print(f"val_fault1_err_mean={fault1.mean().item():.6f} std={fault1.std(unbiased=False).item():.6f}")

    print("training_complete")
    print(f"best_val_loss={best_val:.6f}")
    print(f"threshold={threshold:.6f}")
    print(f"weights_path={weights_path}")
    print(f"threshold_path={threshold_path}")
    print(f"metadata_path={metadata_path}")


if __name__ == "__main__":
    main()
