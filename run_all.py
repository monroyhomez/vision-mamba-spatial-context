#!/usr/bin/env python3
"""
Reproducible experiment runner for the CCC camera-ready revision of
"Overcoming Spatial Context Loss in Vision Mamba".

Runs, in one shot, every empirical number the paper needs and writes them to
./outputs/ as machine-readable CSV/JSON. NOTHING is hardcoded: every value the
paper reports must come from these artifacts.

Blocks:
  [1] Linear-probing accuracy over 3 seeds  -> outputs/accuracy_raw.csv / accuracy_summary.csv
  [2] Hardware profiling (VRAM/latency/FPS) -> outputs/hardware.csv
  [3] Parameter counts                       -> outputs/params.csv
  [4a] Toy directional-scan accuracy ablation-> outputs/ablation_toy.csv
  [4b] Proxy Cross-Scan latency micro-bench   -> outputs/ablation_proxy.csv
  [env] library / GPU versions               -> outputs/env.json

Block [4c] (real VMamba scan ablation) lives in ablation_vmamba_real.py because
it needs the official VMamba repo + selective_scan CUDA kernel.

Usage:
    python run_all.py                 # everything
    python run_all.py --only accuracy hardware   # subset
    python run_all.py --quick         # tiny smoke test (1 seed, 2 epochs)
"""
import argparse
import json
import os
import platform
import random
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CFG = {
    "dataset": "pathmnist",
    "seeds": [42, 123, 2026],
    "probe_epochs": 5,
    "probe_lr": 1e-3,
    "probe_wd": 1e-4,
    "accuracy_resolution": 224,
    "hw_resolutions": [224, 512, 1024],
    "batch_by_res": {224: 64, 512: 8, 1024: 1},
    "warmup_iters": 5,
    "measure_iters": 20,
    "num_workers": 4,
    # toy ablation
    "toy_resolution": 64,
    "toy_epochs": 5,
    "toy_hidden": 64,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def load_vit():
    import timm
    return timm.create_model("vit_base_patch16_224", pretrained=True,
                             num_classes=1000, dynamic_img_size=True).to(DEVICE).eval()


def load_resnet():
    import timm
    return timm.create_model("resnet50", pretrained=True, num_classes=1000).to(DEVICE).eval()


def load_mambavision():
    from mambavision import create_model
    return create_model("mamba_vision_T", pretrained=True).to(DEVICE).eval()


MODELS = {"ViT-Base": load_vit, "ResNet-50": load_resnet, "MambaVision": load_mambavision}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _to3(x):
    return x.repeat(3, 1, 1) if x.shape[0] == 1 else x


def make_transform(size):
    return T.Compose([
        T.Resize((size, size), interpolation=TF.InterpolationMode.BILINEAR, antialias=True),
        T.ToTensor(),
        T.Lambda(_to3),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_loaders(size, batch):
    import medmnist
    from medmnist import INFO
    info = INFO[CFG["dataset"]]
    DataClass = getattr(medmnist, info["python_class"])
    n_classes = len(info["label"])
    tf = make_transform(size)
    tr = DataClass(split="train", transform=tf, download=True)
    te = DataClass(split="test", transform=tf, download=True)
    tl = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=CFG["num_workers"], pin_memory=True)
    vl = DataLoader(te, batch_size=batch, shuffle=False, num_workers=CFG["num_workers"], pin_memory=True)
    return tl, vl, n_classes


# --------------------------------------------------------------------------- #
# [1] Linear probing
# --------------------------------------------------------------------------- #
def replace_head(model, n_classes):
    for p in model.parameters():
        p.requires_grad = False
    for attr in ["head", "fc", "classifier"]:
        if hasattr(model, attr):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Linear):
                new = nn.Linear(mod.in_features, n_classes).to(DEVICE)
                setattr(model, attr, new)
                for p in new.parameters():
                    p.requires_grad = True
                return attr
    raise ValueError(f"No linear head found in {model.__class__.__name__}")


def linear_probe(model_name, seed, epochs):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    set_seed(seed)
    model = MODELS[model_name]()
    tl, vl, n_classes = get_loaders(CFG["accuracy_resolution"], CFG["batch_by_res"][CFG["accuracy_resolution"]])
    attr = replace_head(model, n_classes)
    opt = torch.optim.AdamW(getattr(model, attr).parameters(),
                            lr=CFG["probe_lr"], weight_decay=CFG["probe_wd"])
    crit = nn.CrossEntropyLoss()
    model.eval()
    getattr(model, attr).train()
    for _ in range(epochs):
        for x, y in tl:
            x = x.to(DEVICE, non_blocking=True)
            y = y.view(-1).long().to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
    model.eval()
    yp, yt = [], []
    with torch.no_grad():
        for x, y in vl:
            x = x.to(DEVICE, non_blocking=True)
            yp.extend(model(x).argmax(1).cpu().tolist())
            yt.extend(y.view(-1).cpu().tolist())
    acc = accuracy_score(yt, yp)
    pr, rc, f1, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
    del model
    torch.cuda.empty_cache()
    return {"accuracy": acc * 100, "macro_f1": f1 * 100,
            "precision": pr * 100, "recall": rc * 100}


def run_accuracy(epochs):
    rows = []
    for name in MODELS:
        for seed in CFG["seeds"]:
            print(f"  [accuracy] {name} seed={seed}")
            try:
                m = linear_probe(name, seed, epochs)
                m.update({"model": name, "seed": seed})
                rows.append(m)
            except Exception as e:  # noqa
                print(f"    !! failed: {e}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "accuracy_raw.csv", index=False)
    if not df.empty:
        agg = df.groupby("model").agg(
            acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
            f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
            prec_mean=("precision", "mean"), prec_std=("precision", "std"),
            rec_mean=("recall", "mean"), rec_std=("recall", "std"),
            n=("accuracy", "count")).reset_index()
        agg.to_csv(OUT / "accuracy_summary.csv", index=False)
        print(agg.to_string(index=False))


# --------------------------------------------------------------------------- #
# [2] Hardware profiling
# --------------------------------------------------------------------------- #
def profile_hw(model_name, res, batch):
    torch.cuda.empty_cache()
    try:
        model = MODELS[model_name]()
    except Exception as e:  # noqa
        return {"model": model_name, "resolution": res, "status": f"load_error:{e}"}
    model.eval()
    x = torch.randn(batch, 3, res, res, device=DEVICE)
    try:
        with torch.no_grad():
            for _ in range(CFG["warmup_iters"]):
                model(x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            lat = []
            for _ in range(CFG["measure_iters"]):
                s.record()
                model(x)
                e.record()
                torch.cuda.synchronize()
                lat.append(s.elapsed_time(e))
        peak = torch.cuda.max_memory_allocated() / 1024**2
        lat_img = (sum(lat) / len(lat)) / batch
        out = {"model": model_name, "resolution": res, "batch": batch,
               "peak_vram_mb": peak, "vram_per_image_mb": peak / batch,
               "latency_per_image_ms": lat_img, "throughput_fps": 1000.0 / lat_img,
               "status": "OK"}
    except RuntimeError as err:
        out = {"model": model_name, "resolution": res, "batch": batch,
               "status": "OOM" if "out of memory" in str(err).lower() else f"error:{err}"}
    finally:
        del model, x
        torch.cuda.empty_cache()
    return out


def run_hardware():
    rows = []
    for name in MODELS:
        for res in CFG["hw_resolutions"]:
            b = CFG["batch_by_res"][res]
            print(f"  [hardware] {name} @ {res} (b={b})")
            rows.append(profile_hw(name, res, b))
    pd.DataFrame(rows).to_csv(OUT / "hardware.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


# --------------------------------------------------------------------------- #
# [3] Parameter counts
# --------------------------------------------------------------------------- #
def run_params():
    rows = []
    for name in MODELS:
        try:
            m = MODELS[name]()
            total = sum(p.numel() for p in m.parameters())
            rows.append({"model": name, "params_total": total, "params_millions": total / 1e6})
            del m
            torch.cuda.empty_cache()
        except Exception as e:  # noqa
            rows.append({"model": name, "status": f"error:{e}"})
    pd.DataFrame(rows).to_csv(OUT / "params.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


# --------------------------------------------------------------------------- #
# [4a] Toy directional-scan ablation (controlled, fixed param budget)
# --------------------------------------------------------------------------- #
class DirectionalScanNet(nn.Module):
    """
    Controlled ablation model. A single shared learnable scan cell (1x1 in/out
    projections) aggregates directional context via causal cumulative sums.
    The number of directions (1/2/4) changes ONLY how many oriented passes are
    summed; the parameter count is identical across configurations, so any
    accuracy difference is attributable to directional coverage, not capacity.
    """
    def __init__(self, in_ch=3, hidden=64, n_classes=9, directions=4):
        super().__init__()
        self.directions = directions
        self.embed = nn.Conv2d(in_ch, hidden, kernel_size=3, stride=2, padding=1)
        self.in_proj = nn.Conv2d(hidden, hidden, 1)
        self.out_proj = nn.Conv2d(hidden, hidden, 1)
        self.norm = nn.BatchNorm2d(hidden)
        self.head = nn.Linear(hidden, n_classes)

    def _scan(self, z):
        # z: (B,C,H,W). Causal directional aggregation via cumulative sum.
        B, C, H, W = z.shape
        outs = []
        # 1: row forward
        outs.append(torch.cumsum(z.reshape(B, C, -1), dim=2).reshape(B, C, H, W))
        if self.directions >= 2:  # row backward
            outs.append(torch.flip(torch.cumsum(torch.flip(z, [3]).reshape(B, C, -1), dim=2).reshape(B, C, H, W), [3]))
        if self.directions >= 4:  # column forward + backward
            zt = z.transpose(2, 3)
            cf = torch.cumsum(zt.reshape(B, C, -1), dim=2).reshape(B, C, W, H).transpose(2, 3)
            cb = torch.flip(torch.cumsum(torch.flip(zt, [3]).reshape(B, C, -1), dim=2).reshape(B, C, W, H).transpose(2, 3), [2])
            outs.append(cf)
            outs.append(cb)
        return sum(outs) / len(outs)

    def forward(self, x):
        z = torch.relu(self.embed(x))
        z = self.in_proj(z)
        z = self._scan(z)
        z = torch.relu(self.norm(self.out_proj(z)))
        z = torch.mean(z, dim=(2, 3))
        return self.head(z)


def train_toy(directions, seed, epochs):
    from sklearn.metrics import accuracy_score
    set_seed(seed)
    tl, vl, n_classes = get_loaders(CFG["toy_resolution"], 128)
    model = DirectionalScanNet(hidden=CFG["toy_hidden"], n_classes=n_classes,
                               directions=directions).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    # latency of one forward pass
    model.eval()
    xb = next(iter(vl))[0].to(DEVICE)
    with torch.no_grad():
        for _ in range(5):
            model(xb)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        lat = []
        for _ in range(20):
            s.record(); model(xb); e.record(); torch.cuda.synchronize()
            lat.append(s.elapsed_time(e))
    latency_ms = sum(lat) / len(lat)
    model.train()
    for _ in range(epochs):
        for x, y in tl:
            x = x.to(DEVICE); y = y.view(-1).long().to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward(); opt.step()
    model.eval()
    yp, yt = [], []
    with torch.no_grad():
        for x, y in vl:
            x = x.to(DEVICE)
            yp.extend(model(x).argmax(1).cpu().tolist())
            yt.extend(y.view(-1).cpu().tolist())
    del model; torch.cuda.empty_cache()
    return {"accuracy": accuracy_score(yt, yp) * 100, "params": n_params, "latency_ms": latency_ms}


def run_ablation_toy(epochs):
    rows = []
    for d in [1, 2, 4]:
        for seed in CFG["seeds"]:
            print(f"  [ablation-toy] directions={d} seed={seed}")
            r = train_toy(d, seed, epochs)
            r.update({"directions": d, "seed": seed})
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ablation_toy_raw.csv", index=False)
    agg = df.groupby("directions").agg(
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
        latency_ms=("latency_ms", "mean"), params=("params", "first"),
        n=("accuracy", "count")).reset_index()
    agg.to_csv(OUT / "ablation_toy.csv", index=False)
    print(agg.to_string(index=False))


# --------------------------------------------------------------------------- #
# [4b] Proxy Cross-Scan latency micro-benchmark
# --------------------------------------------------------------------------- #
def run_ablation_proxy():
    B, C, H, W = 16, 64, 56, 56
    x = torch.randn(B, C, H, W, device=DEVICE)
    for _ in range(20):
        torch.cumsum(x.view(B, C, -1), dim=2)
    torch.cuda.synchronize()

    def timed(fn, iters=100):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(iters):
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return sum(ts) / len(ts)

    def scan1():
        return torch.cumsum(x.view(B, C, -1), dim=2)

    def scan4():
        a = torch.cumsum(x.view(B, C, -1), dim=2)
        b = torch.cumsum(torch.flip(x, [2]).view(B, C, -1), dim=2)
        c = torch.cumsum(torch.transpose(x, 2, 3).reshape(B, C, -1), dim=2)
        d = torch.cumsum(torch.flip(torch.transpose(x, 2, 3), [2]).reshape(B, C, -1), dim=2)
        return a + b + c + d

    t1, t4 = timed(scan1), timed(scan4)
    row = {"proxy_1d_ms": t1, "proxy_4d_ms": t4, "overhead_factor": t4 / max(t1, 1e-6)}
    pd.DataFrame([row]).to_csv(OUT / "ablation_proxy.csv", index=False)
    print(row)


# --------------------------------------------------------------------------- #
# env
# --------------------------------------------------------------------------- #
def dump_env():
    info = {"python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    for pkg in ["timm", "medmnist", "mambavision", "numpy", "pandas", "sklearn"]:
        try:
            info[pkg] = __import__(pkg).__version__
        except Exception:  # noqa
            info[pkg] = "n/a"
    try:
        info["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                                      "--format=csv,noheader"]).decode().strip()
    except Exception:  # noqa
        info["nvidia_smi"] = "n/a"
    (OUT / "env.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["accuracy", "hardware", "params", "ablation_toy", "ablation_proxy", "env"])
    ap.add_argument("--quick", action="store_true", help="tiny smoke test")
    args = ap.parse_args()

    if args.quick:
        CFG["seeds"] = [42]
        CFG["probe_epochs"] = 1
        CFG["toy_epochs"] = 1

    steps = args.only or ["env", "params", "hardware", "accuracy", "ablation_toy", "ablation_proxy"]
    print(f"Device: {DEVICE} | steps: {steps}")
    if "env" in steps:
        print("\n== env =="); dump_env()
    if "params" in steps:
        print("\n== params =="); run_params()
    if "hardware" in steps:
        print("\n== hardware =="); run_hardware()
    if "accuracy" in steps:
        print("\n== accuracy =="); run_accuracy(CFG["probe_epochs"])
    if "ablation_toy" in steps:
        print("\n== ablation (toy directional scan) =="); run_ablation_toy(CFG["toy_epochs"])
    if "ablation_proxy" in steps:
        print("\n== ablation (proxy latency) =="); run_ablation_proxy()
    print(f"\nDone. Artifacts in {OUT}")


if __name__ == "__main__":
    main()
