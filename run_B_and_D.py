#!/usr/bin/env python3
"""
Run the missing NDVI experiments (B and D) for the Amazon deforestation project.

Needs the 4-band TIF release of "Planet: Understanding the Amazon from Space"
(train-tif-v2 folder + train_v2.csv). The JPG release has no near-infrared band.

Experiments (same settings as your notebook):
  A  scratch CNN, RGB          B  scratch CNN, RGB + NIR + NDVI
  C  ResNet-34,   RGB          D  ResNet-34,   RGB + NIR + NDVI

Examples
  python run_B_and_D.py --raw /path/to/amazon_data                 # trains B and D, reuses A and C from the notebook run
  python run_B_and_D.py --raw /path/to/amazon_data --exps A B C D  # cleanest: all four from the TIFs (recommended)

At the end it writes  results_to_paste.txt  (and fig_*.png). Send those back.
"""
import argparse, glob, json, os, random, sys, time
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default=os.environ.get("AMAZON_RAW", "/mnt/c/Users/legen/amazon_data"),
                help="folder containing train-tif-v2/*.tif and train_v2.csv (searched recursively)")
ap.add_argument("--root", default=os.environ.get("AMAZON_ROOT_TIF", "/mnt/c/Users/legen/amazon_project_tif"),
                help="output folder for this TIF run (kept separate from your JPG run)")
ap.add_argument("--jpg-root", default=os.environ.get("AMAZON_ROOT", "/mnt/c/Users/legen/amazon_project"),
                help="your original notebook folder; A and C are reused from here if you do not retrain them")
ap.add_argument("--exps", nargs="*", default=["B", "D"], choices=["A", "B", "C", "D"])
ap.add_argument("--epochs", type=int, default=14)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--tile", type=int, default=128)
ap.add_argument("--rebuild-cache", action="store_true")
ap.add_argument("--no-pretrained", action="store_true", help="testing only: skip ImageNet weights download")
ap.add_argument("--subset", type=int, default=0, help="testing only: use only this many tiles")
ap.add_argument("--no-figs", action="store_true")
args = ap.parse_args()

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (fbeta_score, hamming_loss, recall_score, precision_recall_curve,
                             multilabel_confusion_matrix)
import tifffile
import torch, torch.nn as nn, torch.nn.functional as F, torchvision

seed, tileSize = 42, args.tile
root = args.root
cache, runs, figDir = f"{root}/cache", f"{root}/runs", f"{root}/figs"
for d in (root, cache, runs, figDir):
    os.makedirs(d, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
if device != "cuda":
    print("WARNING: no GPU found. Training 4 networks on CPU will take days. Use a CUDA PyTorch build.")
LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.append(s)

# ------------------------------------------------------------------ 1. find the data
tifs = sorted(glob.glob(f"{args.raw}/**/*.tif", recursive=True))
csvs = (glob.glob(f"{args.raw}/**/train_v2.csv", recursive=True) + glob.glob(f"{args.raw}/**/train_classes.csv", recursive=True)
        + glob.glob(f"{args.raw}/**/labels_train_v2.csv", recursive=True))
if len(tifs) < 100 or not csvs:
    sys.exit(f"Could not find the TIF data under {args.raw}: {len(tifs)} .tif files, {len(csvs)} csv files.\n"
             "Download train-tif-v2 and train_v2.csv first (see the Kaggle commands in the instructions).")
tifDir = os.path.dirname(tifs[0])
df = pd.read_csv(csvs[0])
df = df[df["image_name"].map(lambda n: os.path.exists(os.path.join(tifDir, n + ".tif")))].reset_index(drop=True)
if args.subset:
    df = df.sample(args.subset, random_state=seed).reset_index(drop=True)
tagLists = df["tags"].str.split()
labels = sorted({t for tl in tagLists for t in tl})
nLabels = len(labels)
y = np.zeros((len(df), nLabels), dtype=np.uint8)
for i, tl in enumerate(tagLists):
    for t in tl:
        y[i, labels.index(t)] = 1
names = df["image_name"].tolist()
nTiles = len(names)
say(f"Tiles with a TIF: {nTiles} | labels: {nLabels}")

def read_tif(n):
    a = tifffile.imread(os.path.join(tifDir, n + ".tif")).astype(np.float32)
    if a.ndim == 3 and a.shape[0] == 4 and a.shape[-1] != 4:
        a = np.transpose(a, (1, 2, 0))
    return a

# ------------------------------------------------------------------ 2. band order + normalisation + cache
metaPath = f"{cache}/cache_meta.json"
meta_ok = (os.path.exists(metaPath) and not args.rebuild_cache and os.path.exists(f"{cache}/X.npy")
           and json.load(open(metaPath)).get("n") == nTiles and json.load(open(metaPath)).get("tile") == tileSize)
if not meta_ok:
    rng = np.random.RandomState(seed)
    samp = rng.choice(nTiles, min(600, nTiles), replace=False)
    raw = np.stack([read_tif(names[i]) for i in samp])
    ys = y[samp]
    prim = ys[:, labels.index("primary")] == 1
    open_ = (ys[:, labels.index("agriculture")] == 1) | (ys[:, labels.index("bare_ground")] == 1)
    prim, open_ = prim & ~open_, open_ & ~prim
    tileMean = raw.reshape(len(samp), -1, 4).mean(1)
    if prim.sum() and open_.sum():
        ratio = tileMean[open_].mean(0) / tileMean[prim].mean(0)
        bandOrder = "RGBN" if ratio[0] > 1.15 * ratio[2] else "BGRN"
    else:
        bandOrder = "BGRN"
    orders = {"BGRN": dict(b=0, g=1, r=2, nir=3), "RGBN": dict(b=2, g=1, r=0, nir=3)}
    bandMap = orders[bandOrder]
    ndviT = (tileMean[:, bandMap["nir"]] - tileMean[:, bandMap["r"]]) / (tileMean[:, bandMap["nir"]] + tileMean[:, bandMap["r"]] + 1e-6)
    for l in ("primary", "agriculture", "water"):
        m = ys[:, labels.index(l)] == 1
        if m.sum():
            say(f"mean NDVI for {l}: {ndviT[m].mean():.3f}")
    say("Chosen band order:", bandOrder, "(forest NDVI should be highest, water lowest)")
    lo = np.percentile(raw.reshape(-1, 4), 1, axis=0)
    hi = np.percentile(raw.reshape(-1, 4), 99, axis=0)
    json.dump({"order": bandOrder, **bandMap}, open(f"{cache}/order.json", "w"))
    json.dump({"lo": lo.tolist(), "hi": hi.tolist()}, open(f"{cache}/band_stats.json", "w"))
    del raw

    xmm = np.lib.format.open_memmap(f"{cache}/X.npy", mode="w+", dtype=np.float16, shape=(nTiles, 5, tileSize, tileSize))
    def process(i):
        a = read_tif(names[i])
        red, nir = a[..., bandMap["r"]], a[..., bandMap["nir"]]
        ndvi = (nir - red) / (nir + red + 1e-6)
        chans = [np.clip((a[..., src] - lo[src]) / (hi[src] - lo[src] + 1e-6), 0, 1)
                 for src in (bandMap["r"], bandMap["g"], bandMap["b"], bandMap["nir"])]
        chans.append((np.clip(ndvi, -1, 1) + 1) / 2)
        arr = np.stack(chans, -1).astype(np.float32)
        arr = cv2.resize(arr, (tileSize, tileSize), interpolation=cv2.INTER_AREA)
        xmm[i] = np.transpose(arr, (2, 0, 1)).astype(np.float16)
    with ThreadPoolExecutor(8) as ex:
        for k, _ in enumerate(ex.map(process, range(nTiles))):
            if k % 2000 == 0:
                print(f"cached {k}/{nTiles}")
    xmm.flush()
    np.save(f"{cache}/Y.npy", y)
    json.dump(labels, open(f"{cache}/labels.json", "w"))
    json.dump(names, open(f"{cache}/names.json", "w"))
    json.dump({"n": nTiles, "tile": tileSize, "jpg_mode": False, "channels": ["R", "G", "B", "NIR", "NDVI"]}, open(metaPath, "w"))

    freq = y.sum(0)
    strata = np.array([min(np.where(row == 1)[0], key=lambda j: freq[j]) for row in y])
    idx = np.arange(nTiles)
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=strata, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=strata[tmp], random_state=seed)
    np.savez(f"{cache}/splits.npz", train=np.sort(tr), val=np.sort(va), test=np.sort(te))
    xr = np.load(f"{cache}/X.npy", mmap_mode="r")
    pick = np.sort(rng.choice(np.sort(tr), min(3000, len(tr)), replace=False))
    sub = np.asarray(xr[pick]).astype(np.float32)
    np.savez(f"{cache}/stats.npz", mean=sub.mean((0, 2, 3)), std=sub.std((0, 2, 3)) + 1e-6)

xAll = np.load(f"{cache}/X.npy", mmap_mode="r")
y = np.load(f"{cache}/Y.npy")
labels = json.load(open(f"{cache}/labels.json"))
sp = np.load(f"{cache}/splits.npz")
tr, va, te = sp["train"], sp["val"], sp["test"]
st = np.load(f"{cache}/stats.npz")
chanMean, chanStd = st["mean"], st["std"]
say("train/val/test:", len(tr), len(va), len(te))

# ------------------------------------------------------------------ 3. models (identical to the notebook)
def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True), nn.MaxPool2d(2))
class ScratchCNN(nn.Module):
    def __init__(self, cIn, nOut):
        super().__init__()
        self.features = nn.Sequential(block(cIn, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, nOut))
    def forward(self, x):
        return self.head(self.features(x))
def build_model(arch, cIn, pretrained=True):
    if arch == "scratch":
        return ScratchCNN(cIn, nLabels)
    w = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if (pretrained and not args.no_pretrained) else None
    m = torchvision.models.resnet34(weights=w)
    old = m.conv1
    new = nn.Conv2d(cIn, 64, 7, 2, 3, bias=False)
    with torch.no_grad():
        new.weight[:, :3] = old.weight
        if cIn > 3:
            new.weight[:, 3:] = old.weight.mean(1, keepdim=True).repeat(1, cIn - 3, 1, 1)
    m.conv1 = new
    m.fc = nn.Linear(m.fc.in_features, nLabels)
    return m

def dihedral(x, k):
    if k >= 4:
        x = x.flip(3)
    return torch.rot90(x, k % 4, (2, 3))
def augment(x):
    ks = torch.randint(0, 8, (x.size(0),), device=x.device)
    out = torch.empty_like(x)
    for k in range(8):
        m = ks == k
        if m.any():
            out[m] = dihedral(x[m], k)
    return out
@torch.no_grad()
def predict(model, xs, mean, std, tta=False, bs=256):
    model.eval()
    probs = []
    for s in range(0, len(xs), bs):
        xb = (xs[s:s + bs].to(device).float() - mean) / std
        views = range(8) if tta else [0]
        p = 0
        for k in views:
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                p = p + torch.sigmoid(model(dihedral(xb, k)).float())
        probs.append((p / len(views)).cpu().numpy())
    return np.concatenate(probs)
def macro_f2(yy, p, thr=0.2):
    return fbeta_score(yy, (p >= thr).astype(int), beta=2, average="macro", zero_division=0)
def load_split(idx, chans, chunk=2000):
    out = torch.empty((len(idx), len(chans), xAll.shape[2], xAll.shape[3]), dtype=torch.float16)
    for s in range(0, len(idx), chunk):
        out[s:s + chunk] = torch.from_numpy(np.asarray(xAll[idx[s:s + chunk]])[:, chans])
    return out
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

EXPS = {
    "A": ("A_scratch_rgb",      dict(arch="scratch", chans=[0, 1, 2],          lr=3e-3)),
    "B": ("B_scratch_nir_ndvi", dict(arch="scratch", chans=[0, 1, 2, 3, 4],    lr=3e-3)),
    "C": ("C_resnet_rgb",       dict(arch="resnet",  chans=[0, 1, 2],          lr=1e-3)),
    "D": ("D_resnet_nir_ndvi",  dict(arch="resnet",  chans=[0, 1, 2, 3, 4],    lr=1e-3)),
}
epochs, batch = args.epochs, args.batch

def run(name, cfg):
    seed_all(seed)
    chans = cfg["chans"]
    mean = torch.tensor(chanMean[chans], device=device).view(1, -1, 1, 1).float()
    std = torch.tensor(chanStd[chans], device=device).view(1, -1, 1, 1).float()
    xtr, xva, xte = (load_split(i, chans) for i in (tr, va, te))
    ytr, yva, yte = y[tr].astype(np.float32), y[va], y[te]
    ytrT = torch.from_numpy(ytr)
    model = build_model(cfg["arch"], len(chans)).to(device)
    freq = ytr.mean(0)
    posW = torch.tensor(np.clip(np.sqrt((1 - freq) / np.maximum(freq, 1e-6)), 1, 10), dtype=torch.float32)
    crit = nn.BCEWithLogitsLoss(pos_weight=posW.to(device))
    steps = int(np.ceil(len(tr) / batch))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], epochs=epochs, steps_per_epoch=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    w = (ytr * (np.maximum(freq, 1e-6)[None, :] ** -0.5)).max(1)
    w = torch.from_numpy(w / w.sum())
    hist, best, bestState = [], -1.0, None
    for ep in range(epochs):
        model.train(); t0 = time.time()
        order = torch.multinomial(w, len(tr), replacement=True)
        tot = 0.0
        for s in range(0, len(order), batch):
            b = order[s:s + batch]
            xb = augment((xtr[b].to(device).float() - mean) / std)
            yb = ytrT[b].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(xb)
            loss = crit(logits.float(), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(b)
        pv = predict(model, xva, mean, std)
        f2 = macro_f2(yva, pv)
        hist.append(dict(epoch=ep + 1, train_loss=tot / len(tr), val_macro_f2_at_0_2=f2))
        print(f"[{name}] epoch {ep + 1}/{epochs} loss {tot / len(tr):.4f} val macroF2@0.2 {f2:.4f} ({time.time() - t0:.0f}s)")
        if f2 > best:
            best = f2
            bestState = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(bestState)
    torch.save(bestState, f"{runs}/{name}.pt")
    pv, pt = predict(model, xva, mean, std, True), predict(model, xte, mean, std, True)
    np.savez_compressed(f"{runs}/{name}_probs.npz", val_probs=pv, val_y=yva, test_probs=pt, test_y=yte, val_idx=va, test_idx=te)
    json.dump(hist, open(f"{runs}/{name}_history.json", "w"))
    if not args.no_figs:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].plot([h["train_loss"] for h in hist]); ax[0].set_title("Training loss")
        ax[1].plot([h["val_macro_f2_at_0_2"] for h in hist]); ax[1].set_title("Validation macro F2 (threshold 0.2)")
        for a in ax: a.set_xlabel("epoch")
        plt.tight_layout(); plt.savefig(f"{figDir}/fig_curves_{name}.png", dpi=160); plt.close()
    say(f"[{name}] best val macro F2@0.2 = {best:.4f}")
    del xtr, xva, xte, model
    if device == "cuda":
        torch.cuda.empty_cache()

for key in args.exps:
    name, cfg = EXPS[key]
    if os.path.exists(f"{runs}/{name}_probs.npz"):
        print("Skipping", name, "(already trained; delete its files in runs/ to retrain)")
        continue
    run(name, cfg)

# ------------------------------------------------------------------ 4. evaluation
def probs_path(n):
    p = f"{runs}/{n}_probs.npz"
    if os.path.exists(p):
        return p, "TIF run"
    q = f"{args.jpg_root}/runs/{n}_probs.npz"
    if os.path.exists(q):
        return q, "reused from your JPG notebook run"
    return None, None

expNames, src = [], {}
for key in "ABCD":
    n = EXPS[key][0]
    p, how = probs_path(n)
    if not p:
        continue
    d = np.load(p)
    if len(d["test_idx"]) != len(te) or not (d["test_idx"] == te).all():
        say(f"NOTE: {n} from {how} uses a different test split; skipped.")
        continue
    expNames.append(n); src[n] = how
say("Experiments evaluated:", {n: src[n] for n in expNames})
if not expNames:
    sys.exit("No trained experiments found to evaluate. Re-run with --exps B D (or --exps A B C D).")

rareNames = [l for l in ["artisinal_mine", "artisanal_mine", "conventional_mine", "selective_logging", "slash_burn"] if l in labels]
rareIdx = [labels.index(l) for l in rareNames]
def best_thresholds(yy, p, grid=np.arange(0.02, 0.81, 0.01)):
    thr = np.zeros(yy.shape[1])
    for j in range(yy.shape[1]):
        pred = p[:, j:j + 1] >= grid[None, :]
        pos = yy[:, j:j + 1] == 1
        tp = (pred & pos).sum(0); fp = (pred & ~pos).sum(0); fn = (~pred & pos).sum(0)
        thr[j] = grid[(5 * tp / (5 * tp + 4 * fn + fp + 1e-9)).argmax()]
    return thr
def f2_per_label(yy, pred):
    tp = (pred & (yy == 1)).sum(0); fp = (pred & (yy == 0)).sum(0); fn = (~pred & (yy == 1)).sum(0)
    return 5 * tp / (5 * tp + 4 * fn + fp + 1e-9)
def metrics(yy, pred):
    rec = recall_score(yy, pred.astype(int), average=None, zero_division=0)
    return {"macro_F2": fbeta_score(yy, pred, beta=2, average="macro", zero_division=0),
            "samples_F2 (Kaggle metric)": fbeta_score(yy, pred, beta=2, average="samples", zero_division=0),
            "micro_F2": fbeta_score(yy, pred, beta=2, average="micro", zero_division=0),
            "hamming_loss": hamming_loss(yy, pred),
            "rare_recall_mean": float(rec[rareIdx].mean()),
            **{f"recall_{l}": float(rec[labels.index(l)]) for l in rareNames}}

res, store = {}, {}
for n in expNames:
    d = np.load(probs_path(n)[0])
    thr = best_thresholds(d["val_y"], d["val_probs"])
    pred = d["test_probs"] >= thr[None, :]
    res[n] = metrics(d["test_y"], pred)
    store[n] = dict(y=d["test_y"], pred=pred, probs=d["test_probs"], thr=thr)
    json.dump(dict(zip(labels, thr.tolist())), open(f"{runs}/thresholds_{n}.json", "w"))
table = pd.DataFrame(res).T
table.to_csv(f"{figDir}/results_table.csv")
say("\n===== RESULTS TABLE (test set) =====")
say(table.round(4).to_string())

rng2 = np.random.RandomState(0)
def boot_diff(a, b, reps=1000):
    yy, pa, pb = store[a]["y"], store[a]["pred"], store[b]["pred"]
    diffs = []
    for _ in range(reps):
        i = rng2.randint(0, len(yy), len(yy))
        diffs.append(f2_per_label(yy[i], pb[i]).mean() - f2_per_label(yy[i], pa[i]).mean())
    return np.mean(diffs), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
pairs = [("A_scratch_rgb", "B_scratch_nir_ndvi", "NDVI effect, scratch CNN (B - A)"),
         ("C_resnet_rgb", "D_resnet_nir_ndvi", "NDVI effect, ResNet (D - C)"),
         ("A_scratch_rgb", "C_resnet_rgb", "Transfer learning, RGB (C - A)"),
         ("B_scratch_nir_ndvi", "D_resnet_nir_ndvi", "Transfer learning, +NDVI (D - B)")]
rows = []
for a, b, lab in pairs:
    if a in store and b in store:
        m, lo_, hi_ = boot_diff(a, b)
        rows.append(dict(comparison=lab, mean_macro_F2_diff=m, ci_low=lo_, ci_high=hi_,
                         verdict="clear gain" if lo_ > 0 else ("clear loss" if hi_ < 0 else "not significant")))
boot = pd.DataFrame(rows)
if len(boot):
    boot.to_csv(f"{figDir}/bootstrap_differences.csv", index=False)
    say("\n===== BOOTSTRAP (1000 resamples, macro F2 difference) =====")
    say(boot.round(4).to_string(index=False))

say("\n===== PER-LABEL F2 (test) =====")
pl = pd.DataFrame({n: f2_per_label(store[n]["y"], store[n]["pred"]) for n in expNames}, index=labels)
pl.to_csv(f"{figDir}/per_label_f2_all.csv")
say(pl.round(3).to_string())

say("\n===== ERROR BY WEATHER (share of the 17 labels predicted wrong) =====")
wx = [w for w in ["clear", "partly_cloudy", "haze", "cloudy"] if w in labels]
wrows = []
for n in expNames:
    wrong = (store[n]["pred"].astype(int) != store[n]["y"]).mean(1)
    for w in wx:
        m = store[n]["y"][:, labels.index(w)] == 1
        wrows.append(dict(model=n, weather=w, tiles=int(m.sum()), mean_label_error=float(wrong[m].mean())))
werr = pd.DataFrame(wrows)
werr.to_csv(f"{figDir}/error_by_weather.csv", index=False)
say(werr.pivot(index="weather", columns="model", values="mean_label_error").loc[wx].round(4).to_string())

# ------------------------------------------------------------------ 5. figures (never allowed to break the results above)
if not args.no_figs and expNames:
    try:
        fig, ax = plt.subplots(figsize=(9, 4.5)); x = np.arange(len(expNames))
        ax.bar(x - 0.2, [res[n]["macro_F2"] for n in expNames], 0.4, label="Macro F2", color="#3FB6A8")
        ax.bar(x + 0.2, [res[n]["samples_F2 (Kaggle metric)"] for n in expNames], 0.4, label="Kaggle-style F2", color="#7EE081")
        ax.axhline(0.85, color="#FF8A3D", ls="--", label="Target 0.85")
        ax.set_xticks(x); ax.set_xticklabels([n.split("_", 1)[0] for n in expNames]); ax.set_ylim(0, 1); ax.legend()
        ax.set_title("Test results by experiment"); plt.tight_layout(); plt.savefig(f"{figDir}/fig_results.png", dpi=200); plt.close()
        bestN = max(expNames, key=lambda n: res[n]["macro_F2"])
        f2l = f2_per_label(store[bestN]["y"], store[bestN]["pred"]); order = np.argsort(f2l)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh([labels[i] for i in order], f2l[order], color=["#FF8A3D" if i in rareIdx else "#3FB6A8" for i in order])
        ax.set_title(f"Per-label F2: {bestN}"); plt.tight_layout(); plt.savefig(f"{figDir}/fig_per_label_f2.png", dpi=200); plt.close()
        fig, axes = plt.subplots(1, len(rareNames), figsize=(4 * len(rareNames), 3.6), squeeze=False)
        for ax, l in zip(axes[0], rareNames):
            j = labels.index(l)
            for n in expNames:
                p_, r_, _ = precision_recall_curve(store[n]["y"][:, j], store[n]["probs"][:, j]); ax.plot(r_, p_, label=n.split("_", 1)[0])
            ax.set_title(l); ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.legend()
        plt.tight_layout(); plt.savefig(f"{figDir}/fig_pr_rare.png", dpi=200); plt.close()
        werr.pivot(index="weather", columns="model", values="mean_label_error").loc[wx].plot.bar(figsize=(8, 4))
        plt.ylabel("share of the 17 labels predicted wrong"); plt.xticks(rotation=0); plt.tight_layout()
        plt.savefig(f"{figDir}/fig_error_by_weather.png", dpi=200); plt.close()
        cms = multilabel_confusion_matrix(store[bestN]["y"], store[bestN]["pred"].astype(int))
        fig, axes = plt.subplots(3, 6, figsize=(15, 7.5))
        for a in axes.ravel(): a.axis("off")
        for j, a in enumerate(axes.ravel()[:nLabels]):
            a.axis("on"); a.imshow(cms[j], cmap="Greens")
            for (r_, c_), v in np.ndenumerate(cms[j]): a.text(c_, r_, int(v), ha="center", va="center", fontsize=9)
            a.set_title(labels[j], fontsize=9)
        plt.tight_layout(); plt.savefig(f"{figDir}/fig_confusion_matrices.png", dpi=200); plt.close()
    except Exception as e:
        say("Figure step skipped:", repr(e))

    # Grad-CAM: does the NIR+NDVI model fix tiles the RGB model misses?
    try:
        if "C_resnet_rgb" in store and "D_resnet_nir_ndvi" in store and os.path.exists(f"{runs}/D_resnet_nir_ndvi.pt"):
            chansOf = {"C_resnet_rgb": [0, 1, 2], "D_resnet_nir_ndvi": [0, 1, 2, 3, 4]}
            def load_net(n):
                m = torchvision.models.resnet34(weights=None)
                m.conv1 = nn.Conv2d(len(chansOf[n]), 64, 7, 2, 3, bias=False); m.fc = nn.Linear(m.fc.in_features, nLabels)
                m.load_state_dict(torch.load(f"{runs}/{n}.pt", map_location="cpu")); return m.to(device).eval()
            nets = {n: load_net(n) for n in chansOf}
            def gradcam(model, x, j):
                acts, grads = [], []
                def hook(mod, inp, out): acts.append(out); out.register_hook(lambda g: grads.append(g))
                h = model.layer4.register_forward_hook(hook); model.zero_grad(); out = model(x); out[0, j].backward(); h.remove()
                cam = F.relu((grads[0].mean((2, 3), keepdim=True) * acts[0]).sum(1, keepdim=True))
                cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]; cam = cam - cam.min()
                return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()
            def prep(gi, n):
                ch = chansOf[n]; x = torch.from_numpy(np.asarray(xAll[gi])[ch].astype(np.float32))[None].to(device)
                m_ = torch.tensor(chanMean[ch], device=device).view(1, -1, 1, 1).float(); s_ = torch.tensor(chanStd[ch], device=device).view(1, -1, 1, 1).float()
                return (x - m_) / s_
            cD, dD = store["C_resnet_rgb"], store["D_resnet_nir_ndvi"]
            for lab in [l for l in ["selective_logging", "slash_burn", "artisinal_mine", "agriculture"] if l in labels]:
                j = labels.index(lab); truth = cD["y"][:, j] == 1
                fixed = np.where(truth & ~cD["pred"][:, j] & dD["pred"][:, j])[0][:3]
                both = np.where(truth & ~cD["pred"][:, j] & ~dD["pred"][:, j])[0][:2]
                picks = [("D fixes C", i) for i in fixed] + [("both miss", i) for i in both]
                if not picks: continue
                fig, axes = plt.subplots(len(picks), 3, figsize=(9, 2.7 * len(picks)), squeeze=False)
                for r_, (kind, i) in enumerate(picks):
                    gi = int(te[i]); base = np.clip(np.transpose(np.asarray(xAll[gi])[:3].astype(np.float32), (1, 2, 0)) ** 0.8, 0, 1)
                    axes[r_][0].imshow(base); axes[r_][0].set_title(f"{kind} ({lab})", fontsize=8)
                    for c_, n in enumerate(chansOf, start=1):
                        axes[r_][c_].imshow(base); axes[r_][c_].imshow(gradcam(nets[n], prep(gi, n), j), cmap="jet", alpha=0.45)
                        axes[r_][c_].set_title(["RGB-only model", "NIR+NDVI model"][c_ - 1], fontsize=8)
                    for a in axes[r_]: a.axis("off")
                plt.tight_layout(); plt.savefig(f"{figDir}/fig_gradcam_{lab}.png", dpi=200); plt.close()
    except Exception as e:
        say("Grad-CAM step skipped:", repr(e))

# ------------------------------------------------------------------ 6. the file to send back
mixed = any("reused" in v for v in src.values())
with open(f"{root}/results_to_paste.txt", "w") as f:
    f.write("\n".join(LOG))
    if mixed:
        f.write("\n\nNOTE: some rows were reused from the JPG notebook run, so the RGB source differs "
                "from the TIF-based rows. For the cleanest comparison re-run with --exps A B C D.\n")
print("\n" + "=" * 70)
print(f"DONE. Send me this file:  {root}/results_to_paste.txt")
print(f"and the pictures in:      {figDir}  (fig_*.png, *.csv)")
print("=" * 70)
