"""
convmixer_torch.py
------------------
ConvMixer architecture in PyTorch.
Paper: "Patches Are All You Need?" (Trockman & Kolter, TMLR 2022)

Architecture
------------
Input (N, 3, H, W)
→ Conv2D(h, patch_size, stride=patch_size) + GELU + BN   [patch embedding]
→ [DepthwiseConv(kernel_size) + Residual + GELU + BN,
   Conv1×1 + GELU + BN] × depth                          [mixer blocks]
→ AdaptiveAvgPool → Flatten → Linear(n_classes)

Usage
-----
    python models/convmixer_torch.py
    python models/convmixer_torch.py --h 128 --depth 6 --epochs 30
"""

import os
import sys
import csv
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ===========================================================================
# Model
# ===========================================================================

class ConvMixerBlock(nn.Module):
    """
    One ConvMixer mixer block.

    1. Depthwise conv (spatial mixing) + residual + GELU + BN
    2. Pointwise 1×1 conv (channel mixing) + GELU + BN
    """
    def __init__(self, h: int, kernel_size: int):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(h, h, kernel_size=kernel_size,
                      padding=kernel_size // 2, groups=h),
            nn.GELU(),
            nn.BatchNorm2d(h),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(h, h, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(h),
        )

    def forward(self, x):
        x = x + self.depthwise(x)   # residual
        x = self.pointwise(x)
        return x


class ConvMixer(nn.Module):
    """
    ConvMixer model.

    Parameters
    ----------
    h           : int   hidden dimension
    depth       : int   number of mixer blocks
    patch_size  : int   patch embedding kernel/stride
    kernel_size : int   depthwise conv kernel size
    n_classes   : int   output classes
    in_channels : int   input channels (default 3)
    """
    def __init__(self, h=256, depth=8, patch_size=7,
                 kernel_size=9, n_classes=6, in_channels=3):
        super().__init__()

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, h,
                      kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(h),
        )

        self.blocks = nn.Sequential(
            *[ConvMixerBlock(h, kernel_size) for _ in range(depth)]
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(h, n_classes),
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        return self.head(x)


# ===========================================================================
# Training
# ===========================================================================

def train_convmixer(
    processed_dir: str  = "data/processed",
    log_dir:       str  = "logs",
    h:             int  = 256,
    depth:         int  = 8,
    patch_size:    int  = 7,
    kernel_size:   int  = 9,
    n_classes:     int  = 6,
    lr:            float = 1e-3,
    epochs:        int  = 50,
    batch_size:    int  = 64,
    patience:      int  = 10,
    run_name:      str  = None,
    verbose:       bool = True,
):
    if run_name is None:
        run_name = f"convmixer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    os.makedirs(log_dir, exist_ok=True)
    log_path  = os.path.join(log_dir, f"{run_name}_logs.csv")
    ckpt_path = os.path.join(log_dir, f"{run_name}_best.pt")
    cfg_path  = os.path.join(log_dir, f"{run_name}_config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load data ----
    print("Loading data...")
    def load(split):
        p = os.path.join(processed_dir, f"{split}.npz")
        d = np.load(p, allow_pickle=True)
        # (N,H,W,3) → (N,3,H,W)
        X = torch.tensor(d["images"].transpose(0,3,1,2), dtype=torch.float32)
        y = torch.tensor(d["labels"], dtype=torch.long)
        return X, y

    X_train, y_train = load("train_augmented")
    X_val,   y_val   = load("val")
    X_test,  y_test  = load("test")
    print(f"  Train: {tuple(X_train.shape)}  Val: {tuple(X_val.shape)}  Test: {tuple(X_test.shape)}")

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),
                              batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),
                              batch_size=batch_size, shuffle=False, pin_memory=True)

    # ---- Model ----
    model = ConvMixer(h=h, depth=depth, patch_size=patch_size,
                      kernel_size=kernel_size, n_classes=n_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # ---- Optimizer + schedule ----
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    # Save config
    config = dict(model="convmixer", h=h, depth=depth, patch_size=patch_size,
                  kernel_size=kernel_size, lr=lr, epochs=epochs,
                  batch_size=batch_size, patience=patience, n_classes=n_classes)
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    # ---- CSV log ----
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch","train_loss","val_loss","train_acc","val_acc","learning_rate"])

    # ---- Training loop ----
    best_val_loss  = float("inf")
    patience_count = 0
    history = {k: [] for k in
               ("epoch","train_loss","val_loss","train_acc","val_acc","learning_rate")}

    print(f"\nTraining ConvMixer  h={h}  depth={depth}  "
          f"patch={patch_size}  kernel={kernel_size}  device={device}")
    print(f"Logs → {log_path}\n")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss    += loss.item() * len(yb)
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total   += len(yb)

        train_loss /= train_total
        train_acc   = train_correct / train_total

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits  = model(Xb)
                loss    = criterion(logits, yb)
                val_loss    += loss.item() * len(yb)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total   += len(yb)

        val_loss /= val_total
        val_acc   = val_correct / val_total
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Log
        row = [epoch+1, round(train_loss,6), round(val_loss,6),
               round(train_acc,4), round(val_acc,4), round(current_lr,8)]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        for k, v in zip(history.keys(), row):
            history[k].append(v)

        if verbose:
            print(f"  Epoch {epoch+1:>3}/{epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
                  f"lr={current_lr:.2e}")

        # Early stopping + checkpoint
        if val_loss < best_val_loss - 1e-4:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  Early stopping at epoch {epoch+1}.")
                break

    # Restore best
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Test evaluation
    model.eval()
    test_correct, test_total = 0, 0
    all_preds, all_true = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits  = model(Xb)
            preds   = logits.argmax(1)
            test_correct += (preds == yb).sum().item()
            test_total   += len(yb)
            all_preds.append(preds.cpu().numpy())
            all_true.append(yb.cpu().numpy())

    test_acc = test_correct / test_total
    print(f"\n{'='*40}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"{'='*40}")

    result_path = os.path.join(log_dir, f"{run_name}_test_result.json")
    with open(result_path, "w") as f:
        json.dump({"test_acc": test_acc}, f, indent=2)

    return history, model, np.concatenate(all_preds), np.concatenate(all_true)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--h",             type=int,   default=256)
    parser.add_argument("--depth",         type=int,   default=8)
    parser.add_argument("--patch-size",    type=int,   default=7)
    parser.add_argument("--kernel-size",   type=int,   default=9)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch-size",    type=int,   default=64)
    parser.add_argument("--patience",      type=int,   default=10)
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()

    train_convmixer(
        processed_dir=args.processed_dir,
        h=args.h, depth=args.depth,
        patch_size=args.patch_size,
        kernel_size=args.kernel_size,
        lr=args.lr, epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )