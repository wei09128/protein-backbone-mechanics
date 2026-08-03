"""
paper1_table1_gpu_mlp.py — MLP on GPU (PyTorch), shared cache
================================================================================
cuML has no MLPRegressor equivalent, so this uses PyTorch directly, matching
sklearn's MLPRegressor(hidden_layer_sizes=(100,50)) architecture as closely
as reasonable: two hidden layers (100, 50 units), ReLU activations, Adam
optimizer, trained for up to 200 epochs. This is almost certainly the
slowest of the three models on CPU (which is why it's split out here) but
should be fast on GPU.

Usage:
  python paper1_table1_gpu_mlp.py --cache table1_cache.npz
"""

import argparse
import time
import numpy as np
from sklearn.model_selection import GroupKFold

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except Exception as _torch_err:
    # Catch broadly, not just ImportError: a broken CUDA driver linkage,
    # a mismatched libtorch build, or a partial install raises OSError/
    # RuntimeError at import time, not ImportError. Any of those should
    # still fall back to the sklearn CPU path rather than crashing the run.
    print(f"Warning: torch unavailable ({type(_torch_err).__name__}: {_torch_err}); "
          f"falling back to sklearn CPU MLP.")
    TORCH_AVAILABLE = False
    DEVICE = 'cpu'
    from sklearn.neural_network import MLPRegressor


if TORCH_AVAILABLE:
    class MLP(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, 100), nn.ReLU(),
                nn.Linear(100, 50), nn.ReLU(),
                nn.Linear(50, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

def train_torch_mlp(X_train, y_train, X_test, max_epochs=200, seed=0,
                     batch_size=4096, lr=1e-3):
    torch.manual_seed(seed)
    model = MLP(X_train.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    loss_fn = nn.MSELoss()

    y_mean, y_std = y_train.mean(), y_train.std()
    Xt = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor((y_train - y_mean) / y_std, dtype=torch.float32, device=DEVICE)
    Xte = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)

    ds = torch.utils.data.TensorDataset(Xt, yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(max_epochs):
        epoch_loss = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(ds)
        sched.step(epoch_loss)
        if epoch % 20 == 0:
            print(f"    epoch {epoch}: train MSE (z) = {epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        pred_test = model(Xte).cpu().numpy() * y_std + y_mean  # un-standardize
    return pred_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='table1_cache.npz')
    ap.add_argument('--n_splits', type=int, default=5)
    ap.add_argument('--max_epochs', type=int, default=200)
    args = ap.parse_args()

    backend = f'PyTorch ({DEVICE})' if TORCH_AVAILABLE else 'sklearn CPU fallback'
    print(f"Backend: {backend}")

    print(f"Loading {args.cache}...")
    d = np.load(args.cache)
    X_full_sc = d['X_full_sc'].astype(np.float32)
    y_phi = d['y_phi'].astype(np.float32)
    y_psi = d['y_psi'].astype(np.float32)
    groups = d['groups']
    print(f"  {len(y_phi):,} residues, {len(np.unique(groups)):,} groups, "
          f"{X_full_sc.shape[1]} features")

    cv = GroupKFold(n_splits=args.n_splits)
    splits = list(cv.split(X_full_sc, y_phi, groups=groups))

    def cv_predict(X, y):
        oof = np.zeros(len(y), dtype=np.float32)
        for train_idx, test_idx in splits:
            if TORCH_AVAILABLE:
                oof[test_idx] = train_torch_mlp(
                    X[train_idx], y[train_idx], X[test_idx],
                    max_epochs=args.max_epochs)
            else:
                model = MLPRegressor(hidden_layer_sizes=(100, 50),
                                      max_iter=args.max_epochs, random_state=0)
                model.fit(X[train_idx], y[train_idx])
                oof[test_idx] = model.predict(X[test_idx])
        return oof

    t0 = time.time()
    print("Stage 1: predicting phi...")
    phi_oof = cv_predict(X_full_sc, y_phi)
    cv_phi = 1.0 - (np.mean((y_phi - phi_oof) ** 2) / np.var(y_phi))
    print(f"  CV_phi = {cv_phi:.3f}  ({(time.time()-t0)/60:.1f} min elapsed)")

    print("Stage 2: predicting psi...")
    X_aug = np.column_stack([X_full_sc, phi_oof]).astype(np.float32)
    psi_oof = cv_predict(X_aug, y_psi)
    cv_psi = 1.0 - (np.mean((y_psi - psi_oof) ** 2) / np.var(y_psi))

    elapsed = time.time() - t0
    line = f"MLP (2-layer) (full feat, D-ref)\t{cv_phi:.3f}\t{cv_psi:.3f}"
    with open('table1_result_mlp.txt', 'w') as f:
        f.write(line + '\n')
    print(f"\nDone in {elapsed/60:.1f} min. {line}")
    print("Saved table1_result_mlp.txt")


if __name__ == '__main__':
    main()
