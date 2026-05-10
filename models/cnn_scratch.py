"""
cnn_scratch.py
--------------
Convolutional Neural Network implemented entirely from scratch using NumPy.

Architecture (default)
----------------------
    Input  (N, C_in, H, W)   — image batch, channels-first format
    Conv2D  → ReLU → MaxPool
    Conv2D  → ReLU → MaxPool
    Flatten
    FC (fully connected)
    Softmax + Cross-Entropy loss

Math
----
Conv2D forward:
    out[n, f, i, j] = sum_{c,kh,kw} W[f,c,kh,kw] * X[n,c,i+kh,j+kw] + b[f]

Conv2D backward (via im2col):
    dW = sum over output positions of dout * X_col
    dX reconstructed from dout * W via col2im

ReLU:
    forward : out = max(0, x)
    backward: dout * (x > 0)

MaxPool forward:
    out[n,c,i,j] = max of pool window
    backward: route gradient to the max position only

FC forward:
    out = X @ W + b
    backward: dX = dout @ W.T, dW = X.T @ dout, db = sum(dout)

Usage
-----
    from models.cnn_scratch import CNN

    model = CNN(
        input_shape=(3, 150, 150),
        conv_configs=[(16,3,1), (32,3,1)],   # (filters, kernel, stride)
        pool_size=2,
        n_classes=6,
    )
    history = model.fit(X_train, y_train, X_val, y_val,
                        optimizer='adam', epochs=30, batch_size=64)
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


# ===========================================================================
# Low-level primitives
# ===========================================================================

# ---------------------------------------------------------------------------
# im2col / col2im  (efficient convolution via matrix multiply)
# ---------------------------------------------------------------------------

def im2col(X: np.ndarray, kH: int, kW: int,
           stride: int = 1, pad: int = 0) -> np.ndarray:
    """
    Rearrange image patches into columns for efficient convolution.

    Parameters
    ----------
    X      : np.ndarray  (N, C, H, W)
    kH, kW : int         kernel spatial dimensions
    stride : int         convolution stride
    pad    : int         zero-padding on each side

    Returns
    -------
    np.ndarray  (N, C*kH*kW, out_H*out_W)
    """
    N, C, H, W = X.shape
    out_H = (H + 2 * pad - kH) // stride + 1
    out_W = (W + 2 * pad - kW) // stride + 1

    X_pad = np.pad(X, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')

    col = np.zeros((N, C, kH, kW, out_H, out_W), dtype=X.dtype)
    for y in range(kH):
        y_max = y + stride * out_H
        for x in range(kW):
            x_max = x + stride * out_W
            col[:, :, y, x, :, :] = X_pad[:, :, y:y_max:stride, x:x_max:stride]

    # (N, C*kH*kW, out_H*out_W)
    col = col.transpose(0, 1, 2, 3, 4, 5).reshape(N, C * kH * kW, out_H * out_W)
    return col


def col2im(col: np.ndarray, X_shape: tuple, kH: int, kW: int,
           stride: int = 1, pad: int = 0) -> np.ndarray:
    """
    Inverse of im2col — accumulate column gradients back into image shape.

    Parameters
    ----------
    col     : np.ndarray  (N, C*kH*kW, out_H*out_W)
    X_shape : tuple       (N, C, H, W) of the original input
    kH, kW  : int         kernel dimensions
    stride  : int
    pad     : int

    Returns
    -------
    np.ndarray  (N, C, H, W)  gradient w.r.t. input X
    """
    N, C, H, W = X_shape
    out_H = (H + 2 * pad - kH) // stride + 1
    out_W = (W + 2 * pad - kW) // stride + 1

    col_reshaped = col.reshape(N, C, kH, kW, out_H, out_W)
    X_pad = np.zeros((N, C, H + 2*pad, W + 2*pad), dtype=col.dtype)

    for y in range(kH):
        y_max = y + stride * out_H
        for x in range(kW):
            x_max = x + stride * out_W
            X_pad[:, :, y:y_max:stride, x:x_max:stride] += col_reshaped[:, :, y, x, :, :]

    if pad == 0:
        return X_pad
    return X_pad[:, :, pad:-pad, pad:-pad]


# ---------------------------------------------------------------------------
# Conv2D layer
# ---------------------------------------------------------------------------

class Conv2D:
    """
    2D Convolutional layer.

    Parameters
    ----------
    in_channels  : int   number of input channels
    out_channels : int   number of filters (output channels)
    kernel_size  : int   square kernel side length
    stride       : int   convolution stride (default 1)
    pad          : int   zero-padding (default 1 → 'same' for k=3)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, pad: int = 1):
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kH = self.kW = kernel_size
        self.stride = stride
        self.pad    = pad

        # He initialization
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = (np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
                  * scale).astype(np.float32)
        self.b = np.zeros(out_channels, dtype=np.float32)

        self._cache = None

    @property
    def params(self) -> dict:
        return {"W": self.W, "b": self.b}

    @params.setter
    def params(self, d: dict):
        self.W = d["W"]
        self.b = d["b"]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        X : np.ndarray  (N, C, H, W)

        Returns
        -------
        np.ndarray  (N, F, out_H, out_W)
        """
        N, C, H, W = X.shape
        F = self.out_channels
        out_H = (H + 2*self.pad - self.kH) // self.stride + 1
        out_W = (W + 2*self.pad - self.kW) // self.stride + 1

        X_col = im2col(X, self.kH, self.kW, self.stride, self.pad)
        W_col = self.W.reshape(F, -1)           # (F, C*kH*kW)

        # out[n, f, :] = W_col[f] @ X_col[n]
        out = np.tensordot(W_col, X_col, axes=([1],[1]))  # (F, N, out_H*out_W)
        out = out.transpose(1, 0, 2).reshape(N, F, out_H, out_W)
        out += self.b[np.newaxis, :, np.newaxis, np.newaxis]

        self._cache = (X, X_col)
        return out

    def backward(self, dout: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Backward pass.

        Parameters
        ----------
        dout : np.ndarray  (N, F, out_H, out_W)

        Returns
        -------
        dX    : np.ndarray  (N, C, H, W)
        grads : dict        {"W": dW, "b": db}
        """
        X, X_col = self._cache
        N, F, out_H, out_W = dout.shape

        # Gradient w.r.t. bias
        db = dout.sum(axis=(0, 2, 3))

        # Gradient w.r.t. W
        dout_col = dout.reshape(N, F, -1)           # (N, F, out_H*out_W)
        # dW[f] = sum_n dout_col[n,f,:] @ X_col[n,:,:]^T
        dW = np.tensordot(dout_col, X_col, axes=([0,2],[0,2]))  # (F, C*kH*kW)
        dW = dW.reshape(self.W.shape)

        # Gradient w.r.t. X
        W_col = self.W.reshape(F, -1)               # (F, C*kH*kW)
        # dcol[n,:,p] = W_col.T @ dout_col[n,:,p]
        dcol = np.tensordot(W_col, dout_col, axes=([0],[1]))    # (C*kH*kW, N, out_H*out_W)
        dcol = dcol.transpose(1, 0, 2)              # (N, C*kH*kW, out_H*out_W)
        dX = col2im(dcol, X.shape, self.kH, self.kW, self.stride, self.pad)

        return dX, {"W": dW, "b": db}


# ---------------------------------------------------------------------------
# ReLU
# ---------------------------------------------------------------------------

class ReLU:
    """Rectified Linear Unit activation."""

    def __init__(self):
        self._mask = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._mask = X > 0
        return X * self._mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self._mask


# ---------------------------------------------------------------------------
# MaxPool2D
# ---------------------------------------------------------------------------

class MaxPool2D:
    """
    2D Max Pooling layer.

    Parameters
    ----------
    pool_size : int   square pooling window size (default 2)
    stride    : int   pooling stride (default = pool_size → non-overlapping)
    """

    def __init__(self, pool_size: int = 2, stride: int = None):
        self.pool_size = pool_size
        self.stride    = stride if stride is not None else pool_size
        self._cache    = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray  (N, C, H, W)

        Returns
        -------
        np.ndarray  (N, C, out_H, out_W)
        """
        N, C, H, W = X.shape
        p, s = self.pool_size, self.stride
        out_H = (H - p) // s + 1
        out_W = (W - p) // s + 1

        # Reshape into windows — vectorized using stride tricks
        out   = np.zeros((N, C, out_H, out_W), dtype=X.dtype)
        masks = np.zeros_like(X, dtype=bool)

        for i in range(out_H):
            for j in range(out_W):
                window = X[:, :, i*s:i*s+p, j*s:j*s+p]          # (N,C,p,p)
                max_vals = window.max(axis=(2,3), keepdims=True)   # (N,C,1,1)
                out[:, :, i, j] = max_vals[:, :, 0, 0]
                # Store mask for backward
                mask_window = (window == max_vals)
                masks[:, :, i*s:i*s+p, j*s:j*s+p] |= mask_window

        self._cache = (X.shape, masks, out_H, out_W)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Route gradient to the max position in each window.

        Parameters
        ----------
        dout : np.ndarray  (N, C, out_H, out_W)

        Returns
        -------
        np.ndarray  (N, C, H, W)
        """
        X_shape, masks, out_H, out_W = self._cache
        N, C, H, W = X_shape
        p, s = self.pool_size, self.stride

        dX = np.zeros(X_shape, dtype=dout.dtype)
        for i in range(out_H):
            for j in range(out_W):
                window_mask = masks[:, :, i*s:i*s+p, j*s:j*s+p]
                dX[:, :, i*s:i*s+p, j*s:j*s+p] += (
                    dout[:, :, i:i+1, j:j+1] * window_mask
                )
        return dX


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------

class Flatten:
    """Flatten spatial dims for FC layer."""

    def __init__(self):
        self._shape = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout.reshape(self._shape)


# ---------------------------------------------------------------------------
# Fully Connected
# ---------------------------------------------------------------------------

class FC:
    """
    Fully connected (dense) linear layer.

    Parameters
    ----------
    in_dim  : int   input dimensionality
    out_dim : int   output dimensionality (number of classes for last layer)
    """

    def __init__(self, in_dim: int, out_dim: int):
        scale    = np.sqrt(2.0 / in_dim)
        self.W   = (np.random.randn(in_dim, out_dim) * scale).astype(np.float32)
        self.b   = np.zeros(out_dim, dtype=np.float32)
        self._cache = None

    @property
    def params(self) -> dict:
        return {"W": self.W, "b": self.b}

    @params.setter
    def params(self, d: dict):
        self.W = d["W"]
        self.b = d["b"]

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._cache = X
        return X @ self.W + self.b

    def backward(self, dout: np.ndarray) -> tuple[np.ndarray, dict]:
        X = self._cache
        dX = dout @ self.W.T
        dW = X.T @ dout
        db = dout.sum(axis=0)
        return dX, {"W": dW, "b": db}


# ===========================================================================
# Loss
# ===========================================================================

def softmax_crossentropy(logits: np.ndarray,
                         y: np.ndarray,
                         eps: float = 1e-8) -> tuple[float, np.ndarray]:
    """
    Numerically stable softmax + cross-entropy loss.

    Parameters
    ----------
    logits : np.ndarray  (N, C)
    y      : np.ndarray  (N,)   integer class labels
    eps    : float              probability clipping

    Returns
    -------
    loss   : float
    dlogits: np.ndarray  (N, C)  gradient w.r.t. logits
    """
    N = len(y)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp     = np.exp(shifted)
    probs   = exp / exp.sum(axis=1, keepdims=True)

    clipped = np.clip(probs[np.arange(N), y], eps, 1.0)
    loss    = -np.mean(np.log(clipped))

    dlogits = probs.copy()
    dlogits[np.arange(N), y] -= 1
    dlogits /= N

    return loss, dlogits


# ===========================================================================
# CNN model
# ===========================================================================

class CNN:
    """
    Convolutional Neural Network from scratch.

    Architecture
    ------------
    [Conv(16,3) → ReLU → MaxPool(2)] ×
    [Conv(32,3) → ReLU → MaxPool(2)] ×
    Flatten → FC(n_classes)

    Parameters
    ----------
    input_shape  : tuple  (C, H, W)  e.g. (3, 150, 150)
    conv_configs : list of (out_channels, kernel_size, pad)
                   default: [(16, 3, 1), (32, 3, 1)]
    pool_size    : int    max pooling window size (default 2)
    n_classes    : int    number of output classes
    """

    def __init__(
        self,
        input_shape:  tuple = (3, 150, 150),
        conv_configs: list  = None,
        pool_size:    int   = 2,
        n_classes:    int   = 6,
    ):
        if conv_configs is None:
            conv_configs = [(8, 3, 1), (16, 3, 1)]

        self.input_shape = input_shape
        self.n_classes   = n_classes
        self.layers      = []

        # Build conv blocks
        C, H, W = input_shape
        for out_ch, k, p in conv_configs:
            self.layers.append(Conv2D(C, out_ch, k, stride=1, pad=p))
            self.layers.append(ReLU())
            self.layers.append(MaxPool2D(pool_size))
            H = (H - pool_size) // pool_size + 1
            W = (W - pool_size) // pool_size + 1
            C = out_ch

        self.layers.append(Flatten())
        fc_in = C * H * W
        self.layers.append(FC(fc_in, n_classes))

        self._param_layers = [l for l in self.layers
                              if isinstance(l, (Conv2D, FC))]

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Full forward pass.

        Parameters
        ----------
        X : np.ndarray  (N, C, H, W)  channels-first, float32

        Returns
        -------
        logits : np.ndarray  (N, n_classes)
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dlogits: np.ndarray) -> list[dict]:
        """
        Full backward pass.

        Parameters
        ----------
        dlogits : np.ndarray  (N, n_classes)  gradient from loss

        Returns
        -------
        list of grad dicts for each param layer (Conv2D, FC)
        """
        d = dlogits
        grad_list = []

        for layer in reversed(self.layers):
            if isinstance(layer, (Conv2D, FC)):
                d, grads = layer.backward(d)
                grad_list.insert(0, grads)
            else:
                d = layer.backward(d)

        return grad_list

    def predict(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Predict class indices for X in batches (avoids OOM)."""
        preds = []
        for start in range(0, len(X), batch_size):
            logits = self.forward(X[start:start + batch_size])
            preds.append(np.argmax(logits, axis=1))
        return np.concatenate(preds)

    def accuracy(self, X: np.ndarray, y: np.ndarray,
                 batch_size: int = 64) -> float:
        return float(np.mean(self.predict(X, batch_size) == y))

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _get_all_params(self) -> list[dict]:
        return [l.params for l in self._param_layers]

    def _set_all_params(self, param_list: list[dict]) -> None:
        for layer, p in zip(self._param_layers, param_list):
            layer.params = p

    def save(self, path: str) -> None:
        """Save all layer parameters to .npz."""
        flat = {}
        for i, p in enumerate(self._get_all_params()):
            for k, v in p.items():
                flat[f"layer{i}_{k}"] = v
        np.savez(path, **flat)

    def load(self, path: str) -> None:
        """Load parameters from .npz checkpoint."""
        data   = np.load(path)
        n_layers = len(self._param_layers)
        for i, layer in enumerate(self._param_layers):
            layer.params = {
                k: data[f"layer{i}_{k}"]
                for k in layer.params.keys()
            }

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
        epochs:      int   = 30,
        batch_size:  int   = 64,
        l2:          float = 1e-4,
        clip_norm:   float = 5.0,
        lr_schedule: str   = "cosine",
        lr_min:      float = 1e-6,
        patience:    int   = 7,
        log_dir:     str   = "logs",
        run_name:    str   = None,
        verbose:     bool  = True,
    ) -> dict:
        """
        Train the CNN.

        Parameters
        ----------
        X_train, y_train : np.ndarray  (N, C, H, W) and (N,)
        X_val,   y_val   : validation arrays
        optimizer        : 'sgd' or 'adam'
        lr               : initial learning rate
        epochs           : max epochs
        batch_size       : images per mini-batch
        l2               : L2 weight decay
        clip_norm        : gradient clipping norm
        lr_schedule      : 'step','exponential','cosine','plateau'
        lr_min           : minimum lr floor
        patience         : early stopping patience
        log_dir          : directory for logs and checkpoints
        run_name         : prefix for saved files
        verbose          : print epoch summaries

        Returns
        -------
        dict  history: epoch, train_loss, val_loss, train_acc, val_acc, lr
        """
        if run_name is None:
            run_name = f"cnn_{optimizer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        os.makedirs(log_dir, exist_ok=True)
        log_path  = os.path.join(log_dir, f"{run_name}_logs.csv")
        ckpt_path = os.path.join(log_dir, f"{run_name}_best.npz")
        cfg_path  = os.path.join(log_dir, f"{run_name}_config.json")

        config = dict(
            model="cnn_scratch", optimizer=optimizer, lr=lr,
            epochs=epochs, batch_size=batch_size, l2=l2,
            clip_norm=clip_norm, lr_schedule=lr_schedule,
            patience=patience, input_shape=self.input_shape,
            n_classes=self.n_classes,
        )
        with open(cfg_path, "w") as f:
            json.dump(config, f, indent=2)

        # Build one optimizer instance per param layer
        opt_kwargs = dict(lr=lr, l2=l2, clip_norm=clip_norm)
        def make_opt():
            if optimizer == "adam":
                return Adam(**opt_kwargs)
            elif optimizer == "sgd":
                return SGD(**opt_kwargs, momentum=0.9)
            raise ValueError(f"Unknown optimizer '{optimizer}'.")

        opts = [make_opt() for _ in self._param_layers]
        for opt_i, layer in zip(opts, self._param_layers):
            opt_i.init(layer.params)

        # Use first optimizer's lr for scheduler (all share same lr)
        scheduler = LRScheduler(opts[0], mode=lr_schedule, lr_init=lr,
                                epochs=epochs, lr_min=lr_min)
        stopper   = EarlyStopping(patience=patience, verbose=verbose)

        history = {k: [] for k in
                   ("epoch","train_loss","val_loss","train_acc","val_acc","learning_rate")}

        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=list(history.keys())).writeheader()

        N = len(X_train)
        best_param_snapshot = None

        if verbose:
            print(f"\nTraining CNN  optimizer={optimizer}  lr={lr}  "
                  f"epochs={epochs}  batch={batch_size}")
            print(f"Logs → {log_path}\n")

        for epoch in range(epochs):
            # Shuffle
            idx  = np.random.permutation(N)
            X_sh = X_train[idx]
            y_sh = y_train[idx]

            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, N, batch_size):
                Xb = X_sh[start:start + batch_size]
                yb = y_sh[start:start + batch_size]

                # Forward
                logits = self.forward(Xb)
                loss, dlogits = softmax_crossentropy(logits, yb)

                # Backward
                grad_list = self.backward(dlogits)

                # Update each param layer
                current_lr = opts[0].lr
                for opt_i, layer, grads in zip(opts, self._param_layers, grad_list):
                    opt_i.lr       = current_lr   # sync lr across all opts
                    layer.params   = opt_i.step(layer.params, grads)

                epoch_loss += loss
                n_batches  += 1

            train_loss = epoch_loss / n_batches

            # Validation loss (batched to avoid OOM)
            val_losses = []
            for start in range(0, len(X_val), batch_size):
                Xb = X_val[start:start + batch_size]
                yb = y_val[start:start + batch_size]
                logits = self.forward(Xb)
                vl, _  = softmax_crossentropy(logits, yb)
                val_losses.append(vl)
            val_loss = float(np.mean(val_losses))

            train_acc = self.accuracy(X_train, y_train, batch_size)
            val_acc   = self.accuracy(X_val,   y_val,   batch_size)

            # LR schedule
            scheduler.step(epoch, val_loss)
            new_lr = opts[0].lr
            for opt_i in opts[1:]:
                opt_i.lr = new_lr

            # Log
            row = dict(epoch=epoch+1,
                       train_loss=round(train_loss,6), val_loss=round(val_loss,6),
                       train_acc=round(train_acc,4),   val_acc=round(val_acc,4),
                       learning_rate=round(new_lr,8))
            for k, v in row.items():
                history[k].append(v)
            with open(log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=list(history.keys())).writerow(row)

            if verbose:
                print(f"  Epoch {epoch+1:>3}/{epochs}  "
                      f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                      f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
                      f"lr={new_lr:.2e}")

            # Early stopping — snapshot current params
            current_params = [dict(p) for p in self._get_all_params()]
            if stopper(val_loss, {"snap": np.array([0])}):
                # stopper fired — restore best snapshot
                if best_param_snapshot is not None:
                    self._set_all_params(best_param_snapshot)
                    if verbose:
                        print(f"  Restored best weights.")
                break

            if stopper._wait == 0:   # this was a new best
                best_param_snapshot = [
                    {k: v.copy() for k, v in p.items()}
                    for p in self._get_all_params()
                ]

        self.save(ckpt_path)
        if verbose:
            print(f"  Checkpoint → {ckpt_path}")

        return history


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CNN from scratch.")
    parser.add_argument("--optimizer",    default="adam", choices=["sgd","adam"])
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--l2",           type=float, default=1e-4)
    parser.add_argument("--patience",     type=int,   default=7)
    parser.add_argument("--lr-schedule",  default="cosine",
                        choices=["step","exponential","cosine","plateau"])
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()

    print("Loading preprocessed images...")
    train = np.load(os.path.join(args.processed_dir, "train_augmented.npz"),
                    allow_pickle=True)
    val   = np.load(os.path.join(args.processed_dir, "val.npz"),   allow_pickle=True)
    test  = np.load(os.path.join(args.processed_dir, "test.npz"),  allow_pickle=True)

    # (N,H,W,3) → (N,3,H,W)  channels-first
    X_train = train["images"].transpose(0,3,1,2).astype(np.float32)
    y_train = train["labels"].astype(np.int32)
    X_val   = val["images"].transpose(0,3,1,2).astype(np.float32)
    y_val   = val["labels"].astype(np.int32)
    X_test  = test["images"].transpose(0,3,1,2).astype(np.float32)
    y_test  = test["labels"].astype(np.int32)

    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    model = CNN(
        input_shape=(3, X_train.shape[2], X_train.shape[3]),
        conv_configs=[(16, 3, 1), (32, 3, 1)],
        pool_size=2,
        n_classes=6,
    )

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