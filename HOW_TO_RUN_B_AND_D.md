# Running experiments B and D (NIR + NDVI)

Run everything in the same WSL / Linux shell and Python environment you used for the notebook (it already has torch, torchvision, opencv, scikit-learn, pandas, matplotlib). You need an NVIDIA GPU.

## 1. One-time setup

```bash
pip install tifffile kaggle
sudo apt-get install -y p7zip-full
```

Kaggle access:
1. kaggle.com > your profile > Settings > API > **Create New Token** (downloads `kaggle.json`).
2. Put it in place: `mkdir -p ~/.kaggle && mv /path/to/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
3. Open https://www.kaggle.com/competitions/planet-understanding-the-amazon-from-space/rules and click **Join / I Understand and Accept** (downloads fail with 403 until you do).

## 2. Download and unpack the 4-band TIFs (about 20 GB)

```bash
export AMAZON_RAW=/mnt/c/Users/legen/amazon_data      # same folder your notebook uses
mkdir -p "$AMAZON_RAW" && cd "$AMAZON_RAW"

kaggle competitions files -c planet-understanding-the-amazon-from-space      # confirm exact file names
kaggle competitions download -c planet-understanding-the-amazon-from-space -f train-tif-v2.tar.7z
kaggle competitions download -c planet-understanding-the-amazon-from-space -f train_v2.csv.zip

unzip -o train_v2.csv.zip
7z x train-tif-v2.tar.7z && tar -xf train-tif-v2.tar     # creates train-tif-v2/*.tif
ls train-tif-v2 | wc -l                                    # expect 40479
```

If a downloaded name differs from the listing, use the name shown by the `files` command.

## 3. Run the experiments

Put `run_B_and_D.py` anywhere, then:

```bash
# RECOMMENDED: all four models from the same TIF data (cleanest 2x2 comparison)
python run_B_and_D.py --raw "$AMAZON_RAW" --exps A B C D

# FASTER: only B and D; A and C are reused from your earlier JPG run
python run_B_and_D.py --raw "$AMAZON_RAW" --exps B D
```

Defaults match your notebook: 14 epochs, batch 128, 128x128 tiles, seed 42, same split, same losses and thresholds. Output goes to a new folder (`amazon_project_tif`), so your JPG results are never overwritten. It is safe to re-run: finished experiments are skipped, and a stopped run resumes from the cache.

Useful options: `--epochs N`, `--batch N` (lower it if you run out of GPU memory), `--root PATH`, `--rebuild-cache`.

## 4. What to send back

When it finishes it prints `DONE`. Send:
- `amazon_project_tif/results_to_paste.txt` (results table, bootstrap tests, per-label F2, weather errors)
- the picture files in `amazon_project_tif/figs/` (`fig_*.png`) plus the `.csv` files there
- the first lines of the run output showing "mean NDVI for primary / agriculture / water" and "Chosen band order" (the sanity check that the NIR band was read correctly: primary should have the highest NDVI, water the lowest)

With those I can fill B and D into the scorecard, charts, Grad-CAM comparison and the NDVI-vs-RGB conclusion.
