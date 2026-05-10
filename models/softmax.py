"""
softmax.py
----------
Multiclass Softmax Regression trained with mini-batch gradient descent.

Math
----
Forward pass:
    logits = X @ W + b                          (N, C)
    probs  = softmax(logits)                    (N, C)
    loss   = -mean(sum(Y_onehot * log(probs)))  scalar  (cross-entropy)

Numerically stable softmax:
    softmax(z) = exp(z - max(z)) / sum(exp(z - max(z)))

Backward pass:
    dL/dlogits = (probs - Y_onehot) / N
    dL/dW      = X.T @ dL/dlogits
    dL/db      = sum(dL/dlogits, axis=0)

Usage
-----
    from models.softmax import SoftmaxClassifier

    model = SoftmaxClassifier(n_features=200, n_classes=6)
    history = model.fit(X_train, y_train, X_val, y_val,
                        optimizer="adam", epochs=100, batch_size=256)
    y_pred = model.predict(X_test)
"""

import os
import sys
import csv
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimizer.optimizer import SGD, Adam, LRScheduler, EarlyStopping


# ---------------------------------------------------------------------------
# Numerically stable softmax & cross-entropy
# ---------------------------------------------------------------------------

def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Compute row-wise numerically stable softmax.

    Parameters
    ----------
    logits : np.ndarray  (N, C)

    Returns
    -------
    np.ndarray  (N, C)  probabilities summing to 1 along axis=1
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp     = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy_loss(probs: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute mean cross-entropy loss.

    Parameters
    ----------
    probs : np.ndarray  (N, C)  predicted probabilities
    y     : np.ndarray  (N,)    integer class labels
    eps   : float               clipping for numerical stability

    Returns
    -------
    float  mean cross-entropy loss
    """
    N = len(y)
    clipped = np.clip(probs[np.arange(N), y], eps, 1.0)
    return -np.mean(np.log(clipped))


# ---------------------------------------------------------------------------
# Softmax Classifier
# ---------------------------------------------------------------------------

class SoftmaxClassifier:
    """
    Multiclass Softmax Regression from scratch.

    Parameters
    ----------
    n_features : int   dimensionality of input feature vectors
    n_classes  : int   number of output classes
    """

    def __init__(self, n_features: int, n_classes: int):
        self.n_features = n_features
        self.n_classes  = n_classes
        self.params     = {}
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialization for stable training."""
        scale = np.sqrt(2.0 / (self.n_features + self.n_classes))
        self.params = {
            "W": np.random.randn(self.n_features, self.n_classes).astype(np.float32) * scale,
            "b": np.zeros(self.n_classes, dtype=np.float32),
        }

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.

        Parameters
        ----------
        X : np.ndarray  (N, D)

        Returns
        -------
        logits : np.ndarray  (N, C)
        probs  : np.ndarray  (N, C)
        """
        logits = X @ self.params["W"] + self.params["b"]
        probs  = softmax(logits)
        return logits, probs

    def backward(self, X: np.ndarray, probs: np.ndarray,
                 y: np.ndarray) -> dict:
        """
        Backward pass — compute gradients.

        Parameters
        ----------
        X     : np.ndarray  (N, D)  input batch
        probs : np.ndarray  (N, C)  softmax probabilities
        y     : np.ndarray  (N,)    integer labels

        Returns
        -------
        dict  {"W": dW, "b": db}
        """
        N = len(y)
        d_logits = probs.copy()
        d_logits[np.arange(N), y] -= 1
        d_logits /= N

        dW = X.T @ d_logits
        db = d_logits.sum(axis=0)

        return {"W": dW, "b": db}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probabilities for X."""
        _, probs = self.forward(X)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices for X."""
        return np.argmax(self.predict_proba(X), axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy on (X, y)."""
        return float(np.mean(self.predict(X) == y))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        optimizer:   str   = "adam",
        lr:          float = 1e-3,
        epochs:      int   = 100,
        batch_size:  int   = 256,
        l2:          float = 1e-4,
        clip_norm:   float = 5.0,
        lr_schedule: str   = "cosine",
        lr_min:      float = 1e-6,
        step_size:   int   = 10,
        gamma:       float = 0.5,
        patience:    int   = 10,
        log_dir:     str   = "logs",
        run_name:    str   = None,
        verbose:     bool  = True,
    ) -> dict:
        """
        Train the softmax classifier.

        Parameters
        ----------
        X_train, y_train : training data  (N_tr, D) and (N_tr,)
        X_val,   y_val   : validation data
        optimizer        : 'sgd' or 'adam'
        lr               : initial learning rate
        epochs           : maximum training epochs
        batch_size       : mini-batch size
        l2               : L2 regularization coefficient
        clip_norm        : gradient clipping norm threshold
        lr_schedule      : 'step', 'exponential', 'cosine', or 'plateau'
        lr_min           : minimum learning rate floor
        step_size        : epochs between step-decay reductions
        gamma            : decay factor for step/exponential schedules
        patience         : early stopping patience (epochs)
        log_dir          : directory to write logs.csv and checkpoint
        run_name         : name prefix for log files (auto-generated if None)
        verbose          : print per-epoch summary

        Returns
        -------
        dict  training history with keys:
              epoch, train_loss, val_loss, train_acc, val_acc, learning_rate
        """
        # ---- Setup ----
        if run_name is None:
            run_name = f"softmax_{optimizer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        os.makedirs(log_dir, exist_ok=True)
        log_path  = os.path.join(log_dir, f"{run_name}_logs.csv")
        ckpt_path = os.path.join(log_dir, f"{run_name}_best.npz")
        cfg_path  = os.path.join(log_dir, f"{run_name}_config.json")

        # Save run config
        config = dict(
            model="softmax", optimizer=optimizer, lr=lr, epochs=epochs,
            batch_size=batch_size, l2=l2, clip_norm=clip_norm,
            lr_schedule=lr_schedule, patience=patience,
            n_features=self.n_features, n_classes=self.n_classes,
        )
        with open(cfg_path, "w") as f:
            json.dump(config, f, indent=2)

        # Build optimizer
        opt_kwargs = dict(lr=lr, l2=l2, clip_norm=clip_norm)
        if optimizer == "sgd":
            opt = SGD(**opt_kwargs, momentum=0.9)
        elif optimizer == "adam":
            opt = Adam(**opt_kwargs)
        else:
            raise ValueError(f"optimizer must be 'sgd' or 'adam', got '{optimizer}'.")

        opt.init(self.params)

        scheduler = LRScheduler(
            opt, mode=lr_schedule, lr_init=lr, epochs=epochs,
            step_size=step_size, gamma=gamma, lr_min=lr_min,
        )
        stopper = EarlyStopping(patience=patience, verbose=verbose)

        history = {k: [] for k in
                   ("epoch", "train_loss", "val_loss", "train_acc", "val_acc", "learning_rate")}

        # Open log CSV
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history.keys()))
            writer.writeheader()

        N = len(X_train)

        if verbose:
            print(f"\nTraining Softmax  optimizer={optimizer}  lr={lr}  "
                  f"epochs={epochs}  batch={batch_size}  l2={l2}")
            print(f"Logs → {log_path}\n")

        for epoch in range(epochs):
            # Mini-batch shuffling
            idx = np.random.permutation(N)
            X_sh, y_sh = X_train[idx], y_train[idx]

            # Mini-batch loop
            epoch_loss = 0.0
            n_batches  = 0
            for start in range(0, N, batch_size):
                Xb = X_sh[start:start + batch_size]
                yb = y_sh[start:start + batch_size]

                _, probs = self.forward(Xb)
                loss     = cross_entropy_loss(probs, yb)
                grads    = self.backward(Xb, probs, yb)

                self.params = opt.step(self.params, grads)
                epoch_loss += loss
                n_batches  += 1

            train_loss = epoch_loss / n_batches

            # Validation
            _, val_probs = self.forward(X_val)
            val_loss     = cross_entropy_loss(val_probs, y_val)
            train_acc    = self.accuracy(X_train, y_train)
            val_acc      = self.accuracy(X_val,   y_val)
            current_lr   = opt.lr

            # LR schedule
            scheduler.step(epoch, val_loss)

            # Log
            row = dict(epoch=epoch + 1, train_loss=round(train_loss, 6),
                       val_loss=round(val_loss, 6),
                       train_acc=round(train_acc, 4), val_acc=round(val_acc, 4),
                       learning_rate=round(current_lr, 8))
            for k, v in row.items():
                history[k].append(v)

            with open(log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=list(history.keys())).writerow(row)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:>4}/{epochs}  "
                      f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                      f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
                      f"lr={current_lr:.2e}")

            # Early stopping
            if stopper(val_loss, self.params):
                break

        # Restore best weights
        if stopper.best_params is not None:
            self.params = stopper.best_params
            if verbose:
                print(f"\n  Restored best weights (val_loss={stopper._best_loss:.4f})")

        # Save best checkpoint
        np.savez(ckpt_path, **self.params)
        if verbose:
            print(f"  Checkpoint → {ckpt_path}")

        return history

    def load(self, path: str) -> None:
        """Load parameters from a .npz checkpoint file."""
        data = np.load(path)
        self.params = {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# CLI — train and evaluate
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Softmax classifier.")
    parser.add_argument("--optimizer",   default="adam", choices=["sgd", "adam"])
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch-size",  type=int,   default=256)
    parser.add_argument("--l2",          type=float, default=1e-4)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--lr-schedule", default="cosine",
                        choices=["step", "exponential", "cosine", "plateau"])
    parser.add_argument("--features-dir", default="features")
    parser.add_argument("--use-mrmr",    action="store_true",
                        help="Use MRMR-selected features instead of full features")
    args = parser.parse_args()

    suffix = "mrmr" if args.use_mrmr else "features"

    print("Loading features...")
    train = np.load(os.path.join(args.features_dir, f"train_{suffix}.npz"))
    val   = np.load(os.path.join(args.features_dir, f"val_{suffix}.npz"))
    test  = np.load(os.path.join(args.features_dir, f"test_{suffix}.npz"))

    X_train, y_train = train["X"].astype(np.float32), train["y"]
    X_val,   y_val   = val["X"].astype(np.float32),   val["y"]
    X_test,  y_test  = test["X"].astype(np.float32),  test["y"]

    # Feature normalization (z-score, fit on train only)
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std

    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    model = SoftmaxClassifier(n_features=X_train.shape[1], n_classes=6)

    history = model.fit(
        X_train, y_train, X_val, y_val,
        optimizer=args.optimizer,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        l2=args.l2,
        patience=args.patience,
        lr_schedule=args.lr_schedule,
    )

    test_acc = model.accuracy(X_test, y_test)
    print(f"\n{'='*40}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"{'='*40}")