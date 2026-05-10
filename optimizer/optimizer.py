"""
optimizer.py
------------
Optimizer implementations for Softmax Regression and CNN-from-scratch.

Optimizers
----------
    SGD       — stochastic gradient descent (baseline)
    Adam      — adaptive moment estimation (advanced)

Features (both optimizers)
--------------------------
    - Learning rate schedules: step decay, exponential, cosine, reduce-on-plateau
    - Gradient clipping (by global norm)
    - L2 regularization (weight decay) applied to gradients before update
    - Mini-batch shuffling each epoch
    - Early stopping with configurable patience

Usage
-----
    from optimizer.optimizer import Adam, EarlyStopping, LRScheduler

    opt = Adam(lr=1e-3, l2=1e-4)
    opt.init(params)                    # params: dict {name: np.ndarray}
    grads = {"W": dW, "b": db}
    params = opt.step(params, grads)

    scheduler = LRScheduler(opt, mode="cosine", epochs=50)
    scheduler.step(epoch, val_loss)

    stopper = EarlyStopping(patience=7)
    if stopper(val_loss):
        break   # stop training
"""

import numpy as np


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------

def clip_gradients(grads: dict, max_norm: float = 5.0) -> dict:
    """
    Clip gradients by global L2 norm.

    If the global norm of all gradients exceeds max_norm, all gradients
    are scaled down uniformly so the global norm equals max_norm.
    If the norm is within bounds, gradients are returned unchanged.

    Parameters
    ----------
    grads    : dict {name: np.ndarray}   gradient arrays
    max_norm : float                     clipping threshold (default 5.0)

    Returns
    -------
    dict  clipped gradient arrays (same keys)
    """
    global_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
    if global_norm > max_norm:
        scale = max_norm / (global_norm + 1e-8)
        grads = {k: v * scale for k, v in grads.items()}
    return grads


# ---------------------------------------------------------------------------
# SGD
# ---------------------------------------------------------------------------

class SGD:
    """
    Stochastic Gradient Descent with optional momentum.

    Update rule (no momentum):
        param = param - lr * grad

    Update rule (with momentum, mu > 0):
        velocity = mu * velocity - lr * grad
        param    = param + velocity

    Parameters
    ----------
    lr       : float   Learning rate (default 1e-2).
    momentum : float   Momentum coefficient in [0, 1). 0 = plain SGD.
    l2       : float   L2 regularization coefficient (weight decay).
    clip_norm: float   Max gradient norm. 0 = no clipping.
    """

    def __init__(self, lr=1e-2, momentum=0.0, l2=0.0, clip_norm=5.0):
        self.lr        = lr
        self.momentum  = momentum
        self.l2        = l2
        self.clip_norm = clip_norm
        self._velocity = {}

    def init(self, params: dict) -> None:
        """Initialize velocity buffers to zero for all parameters."""
        self._velocity = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict, grads: dict) -> dict:
        """
        Perform one parameter update.

        Parameters
        ----------
        params : dict {name: np.ndarray}   current parameter arrays
        grads  : dict {name: np.ndarray}   gradient arrays (same keys)

        Returns
        -------
        dict   updated parameter arrays
        """
        # L2 gradient contribution
        if self.l2 > 0:
            grads = {k: g + self.l2 * params[k] for k, g in grads.items()}

        # Gradient clipping
        if self.clip_norm > 0:
            grads = clip_gradients(grads, self.clip_norm)

        updated = {}
        for k in params:
            if self.momentum > 0:
                self._velocity[k] = (self.momentum * self._velocity[k]
                                     - self.lr * grads[k])
                updated[k] = params[k] + self._velocity[k]
            else:
                updated[k] = params[k] - self.lr * grads[k]

        return updated


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------

class Adam:
    """
    Adam optimizer (Kingma & Ba, 2015).

    Update rule:
        m = beta1 * m + (1 - beta1) * grad          # first moment
        v = beta2 * v + (1 - beta2) * grad^2         # second moment
        m_hat = m / (1 - beta1^t)                    # bias correction
        v_hat = v / (1 - beta2^t)
        param = param - lr * m_hat / (sqrt(v_hat) + eps)

    Parameters
    ----------
    lr       : float   Learning rate (default 1e-3).
    beta1    : float   First moment decay (default 0.9).
    beta2    : float   Second moment decay (default 0.999).
    eps      : float   Numerical stability (default 1e-8).
    l2       : float   L2 regularization coefficient.
    clip_norm: float   Max gradient norm. 0 = no clipping.
    """

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, l2=0.0, clip_norm=5.0):
        self.lr        = lr
        self.beta1     = beta1
        self.beta2     = beta2
        self.eps       = eps
        self.l2        = l2
        self.clip_norm = clip_norm
        self._m  = {}
        self._v  = {}
        self._t  = 0

    def init(self, params: dict) -> None:
        """Initialize first and second moment buffers to zero."""
        self._m = {k: np.zeros_like(v) for k, v in params.items()}
        self._v = {k: np.zeros_like(v) for k, v in params.items()}
        self._t = 0

    def step(self, params: dict, grads: dict) -> dict:
        """
        Perform one Adam parameter update.

        Parameters
        ----------
        params : dict {name: np.ndarray}
        grads  : dict {name: np.ndarray}

        Returns
        -------
        dict   updated parameter arrays
        """
        self._t += 1

        # L2 gradient contribution
        if self.l2 > 0:
            grads = {k: g + self.l2 * params[k] for k, g in grads.items()}

        # Gradient clipping
        if self.clip_norm > 0:
            grads = clip_gradients(grads, self.clip_norm)

        updated = {}
        for k in params:
            self._m[k] = self.beta1 * self._m[k] + (1 - self.beta1) * grads[k]
            self._v[k] = self.beta2 * self._v[k] + (1 - self.beta2) * grads[k] ** 2

            m_hat = self._m[k] / (1 - self.beta1 ** self._t)
            v_hat = self._v[k] / (1 - self.beta2 ** self._t)

            updated[k] = params[k] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return updated


# ---------------------------------------------------------------------------
# Learning Rate Scheduler
# ---------------------------------------------------------------------------

class LRScheduler:
    """
    Learning rate schedule wrapper for SGD or Adam optimizers.

    Modes
    -----
    step        : multiply lr by gamma every step_size epochs
                  lr = lr_init * gamma^(epoch // step_size)
    exponential : lr = lr_init * gamma^epoch
    cosine      : lr = lr_min + 0.5*(lr_init-lr_min)*(1 + cos(pi*epoch/epochs))
    plateau     : reduce lr by gamma if val_loss hasn't improved in patience epochs

    Parameters
    ----------
    optimizer  : SGD or Adam instance
    mode       : str    one of 'step', 'exponential', 'cosine', 'plateau'
    lr_init    : float  initial learning rate (read from optimizer if None)
    epochs     : int    total epochs (required for cosine mode)
    step_size  : int    epoch interval for step decay (default 10)
    gamma      : float  decay factor (default 0.5)
    lr_min     : float  minimum learning rate floor (default 1e-6)
    patience   : int    plateau patience in epochs (default 5)
    """

    def __init__(self, optimizer, mode="cosine", lr_init=None,
                 epochs=50, step_size=10, gamma=0.5,
                 lr_min=1e-6, patience=5):
        self.opt       = optimizer
        self.mode      = mode
        self.lr_init   = lr_init if lr_init is not None else optimizer.lr
        self.epochs    = epochs
        self.step_size = step_size
        self.gamma     = gamma
        self.lr_min    = lr_min
        self.patience  = patience

        # Plateau state
        self._best_loss    = np.inf
        self._plateau_wait = 0

        if mode not in ("step", "exponential", "cosine", "plateau"):
            raise ValueError(f"mode must be one of step/exponential/cosine/plateau, got '{mode}'.")

    def step(self, epoch: int, val_loss: float = None) -> float:
        """
        Update the optimizer's learning rate for the given epoch.

        Parameters
        ----------
        epoch    : int     current epoch (0-indexed)
        val_loss : float   required only for plateau mode

        Returns
        -------
        float  new learning rate
        """
        if self.mode == "step":
            new_lr = self.lr_init * (self.gamma ** (epoch // self.step_size))

        elif self.mode == "exponential":
            new_lr = self.lr_init * (self.gamma ** epoch)

        elif self.mode == "cosine":
            import math
            new_lr = (self.lr_min +
                      0.5 * (self.lr_init - self.lr_min) *
                      (1 + math.cos(math.pi * epoch / self.epochs)))

        elif self.mode == "plateau":
            if val_loss is None:
                raise ValueError("val_loss required for plateau mode.")
            if val_loss < self._best_loss - 1e-6:
                self._best_loss    = val_loss
                self._plateau_wait = 0
            else:
                self._plateau_wait += 1
            if self._plateau_wait >= self.patience:
                new_lr = max(self.opt.lr * self.gamma, self.lr_min)
                self._plateau_wait = 0
            else:
                new_lr = self.opt.lr

        new_lr = max(new_lr, self.lr_min)
        self.opt.lr = new_lr
        return new_lr


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Early stopping monitor based on validation loss.

    Stops training when validation loss has not improved by more than
    `min_delta` for `patience` consecutive epochs.

    Parameters
    ----------
    patience  : int    epochs to wait without improvement (default 7)
    min_delta : float  minimum improvement to count as progress (default 1e-4)
    verbose   : bool   print message when triggered

    Usage
    -----
        stopper = EarlyStopping(patience=7)
        for epoch in range(max_epochs):
            ...
            if stopper(val_loss):
                print("Early stopping triggered.")
                break
        best_params = stopper.best_params
    """

    def __init__(self, patience=7, min_delta=1e-4, verbose=True):
        self.patience   = patience
        self.min_delta  = min_delta
        self.verbose    = verbose
        self._best_loss = np.inf
        self._wait      = 0
        self.best_params: dict | None = None

    def __call__(self, val_loss: float, params: dict = None) -> bool:
        """
        Check whether to stop training.

        Parameters
        ----------
        val_loss : float   current epoch validation loss
        params   : dict    current model params to save if this is the best epoch

        Returns
        -------
        bool  True = stop training, False = continue
        """
        if val_loss < self._best_loss - self.min_delta:
            self._best_loss = val_loss
            self._wait      = 0
            if params is not None:
                self.best_params = {k: v.copy() for k, v in params.items()}
        else:
            self._wait += 1
            if self._wait >= self.patience:
                if self.verbose:
                    print(f"  Early stopping: no improvement for {self.patience} epochs.")
                return True
        return False