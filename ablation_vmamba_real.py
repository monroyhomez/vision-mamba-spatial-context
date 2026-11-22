#!/usr/bin/env python3
"""
[4c] REAL VMamba directional-scan ablation (optional, heavy setup).

Unlike the toy ablation in run_all.py, this script drives the *official* VMamba
implementation, whose SS2D block performs the genuine selective-scan CUDA kernel
over K oriented trajectories (CrossScan). By overriding K (1 -> 2 -> 4) we isolate
the contribution of true multidirectional routing on a topology-sensitive task.

REQUIREMENTS (do this once on the GPU machine before running):
  1. git clone https://github.com/MzeroMiko/VMamba
  2. cd VMamba/kernels/selective_scan && pip install .        # builds the CUDA kernel
  3. pip install einops timm triton
  4. export VMAMBA_REPO=/absolute/path/to/VMamba

Then:
    python ablation_vmamba_real.py

If the repo/kernel is not available the script SKIPS gracefully and writes a
status note to outputs/ablation_vmamba_real.csv so the run never crashes.

NOTE: The exact attribute name that controls the number of scan directions has
changed across VMamba commits (`k_group`, `directions`, or the hardcoded
CrossScan). If the automatic override below fails, follow the printed hint to
set it manually for your checked-out commit.
"""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 123, 2026]
EPOCHS = 5
RES = 224
BATCH = 64


def skip(reason):
    print(f"[SKIP] {reason}")
    pd.DataFrame([{"status": "skipped", "reason": reason}]).to_csv(
        OUT / "ablation_vmamba_real.csv", index=False)
    sys.exit(0)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_vmamba(num_directions):
    """
    Build a VMamba-Tiny and override the number of scan directions.
    Returns (model, actual_directions_applied).
    """
    repo = os.environ.get("VMAMBA_REPO")
    if not repo or not Path(repo).exists():
        skip("VMAMBA_REPO not set or path missing. See header for setup steps.")
    sys.path.insert(0, str(Path(repo) / "classification"))
    try:
        from models.vmamba import VSSM  # noqa
    except Exception as e:  # noqa
        skip(f"Could not import official VMamba VSSM: {e}")

    model = VSSM(depths=[2, 2, 5, 2], dims=96, num_classes=1000).to(DEVICE)

    # --- Override the number of directions on every SS2D block ---------------
    applied = None
    for m in model.modules():
        for attr in ("k_group", "directions", "n_directions"):
            if hasattr(m, attr):
                try:
                    setattr(m, attr, num_directions)
                    applied = attr
                except Exception:  # noqa
                    pass
    if applied is None:
        print("[HINT] No exposed directions attribute found. For your VMamba "
              "commit, edit models/vmamba.py CrossScan/SS2D to use "
              f"{num_directions} directions, or patch it here.")
    return model.eval(), (applied or "hardcoded-4")


def get_loaders():
    import medmnist
    from medmnist import INFO
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    info = INFO["pathmnist"]
    DataClass = getattr(medmnist, info["python_class"])
    n = len(info["label"])
    tf = T.Compose([
        T.Resize((RES, RES), interpolation=TF.InterpolationMode.BILINEAR, antialias=True),
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    from torch.utils.data import DataLoader
    tr = DataClass(split="train", transform=tf, download=True)
    te = DataClass(split="test", transform=tf, download=True)
    return (DataLoader(tr, BATCH, shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(te, BATCH, shuffle=False, num_workers=4, pin_memory=True), n)


def probe(num_directions, seed):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    set_seed(seed)
    model, applied = load_vmamba(num_directions)
    tl, vl, n_classes = get_loaders()
    # linear probe: freeze all, replace classifier head
    for p in model.parameters():
        p.requires_grad = False
    head_attr = None
    for attr in ("head", "classifier", "fc"):
        if hasattr(model, attr) and isinstance(getattr(model, attr), nn.Linear):
            head_attr = attr
            break
    if head_attr is None:
        skip("Could not locate VMamba linear head.")
    in_f = getattr(model, head_attr).in_features
    setattr(model, head_attr, nn.Linear(in_f, n_classes).to(DEVICE))
    for p in getattr(model, head_attr).parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(getattr(model, head_attr).parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    model.eval(); getattr(model, head_attr).train()
    for _ in range(EPOCHS):
        for x, y in tl:
            x = x.to(DEVICE); y = y.view(-1).long().to(DEVICE)
            opt.zero_grad(set_to_none=True)
            crit(model(x), y).backward(); opt.step()
    model.eval()
    yp, yt = [], []
    with torch.no_grad():
        for x, y in vl:
            yp.extend(model(x.to(DEVICE)).argmax(1).cpu().tolist())
            yt.extend(y.view(-1).cpu().tolist())
    acc = accuracy_score(yt, yp) * 100
    _, _, f1, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
    del model; torch.cuda.empty_cache()
    return acc, f1 * 100, applied


def main():
    rows = []
    for d in [1, 2, 4]:
        for seed in SEEDS:
            print(f"[vmamba-real] directions={d} seed={seed}")
            acc, f1, applied = probe(d, seed)
            rows.append({"directions": d, "seed": seed, "accuracy": acc,
                         "macro_f1": f1, "override_attr": applied})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ablation_vmamba_real_raw.csv", index=False)
    agg = df.groupby("directions").agg(
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
        f1_mean=("macro_f1", "mean"), n=("accuracy", "count")).reset_index()
    agg.to_csv(OUT / "ablation_vmamba_real.csv", index=False)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
