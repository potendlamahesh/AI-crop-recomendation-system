"""
train_model_v2.py - retrain the AI Crop Recommendation model with MORE CROPS.

What it does
  1. Loads the original Kaggle data (Crop_recommendation.csv, 22 real-dataset crops).
  2. Adds extra crops (chilli, tomato, onion, wheat, sugarcane, ...) from
     extra_crop_ranges.csv.  These rows are GENERATED from documented agronomic
     ranges - they are NOT field measurements - and are labelled as such.
  3. Evaluates honestly with 5-fold stratified cross-validation, reporting the
     real-data crops and the range-based crops SEPARATELY, plus robustness to
     noisy (real-world) inputs and top-3 accuracy.
  4. Trains the final Random Forest and saves crop_model.pkl in the SAME format
     the app already loads: {"model", "features", "model_name", ...}.

Run:  python train_model_v2.py            (from the project folder)
Add a crop later: append one row to extra_crop_ranges.csv and re-run.
Safe to re-run: any extra-crop rows already in Crop_recommendation.csv are
dropped and regenerated, never duplicated.
"""
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
DATA_CSV = HERE / "Crop_recommendation.csv"
RANGES_CSV = HERE / "extra_crop_ranges.csv"
MODEL_PKL = HERE / "crop_model.pkl"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
SAMPLES_PER_CROP = 100          # same balance as the original dataset
SEED = 42
NOISE_LEVELS = (0.10, 0.25)     # test-time noise as a fraction of each feature's std
N_TREES = 200                  # keeps crop_model.pkl small enough for GitHub / Streamlit Cloud
MIN_LEAF = 3
AUG_NOISE = 0.15                # training-time noise (fraction of std)
AUG_COPIES = 2


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_real_data():
    df = pd.read_csv(DATA_CSV)
    missing = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing:
        sys.exit(f"Crop_recommendation.csv is missing columns: {missing}")
    return df[FEATURES + ["label"]].copy()


def generate_range_based(ranges: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Truncated-normal samples centred in each documented [min, max] range."""
    rows = []
    for _, r in ranges.iterrows():
        cols = {}
        for f in FEATURES:
            lo, hi = float(r[f + "_min"]), float(r[f + "_max"])
            mid, sd = (lo + hi) / 2, (hi - lo) / 4
            x = rng.normal(mid, sd, SAMPLES_PER_CROP)
            bad = (x < lo) | (x > hi)
            while bad.any():                      # resample instead of clipping
                x[bad] = rng.normal(mid, sd, bad.sum())
                bad = (x < lo) | (x > hi)
            # N, P and K are whole numbers in the original dataset; keep that style
            cols[f] = np.round(x).astype(int) if f in ("N", "P", "K") else np.round(x, 2)
        cols["label"] = r["crop"].strip().lower()
        rows.append(pd.DataFrame(cols))
    return pd.concat(rows, ignore_index=True)


def build_dataset():
    real = load_real_data()
    ranges = pd.read_csv(RANGES_CSV)
    extra_names = set(ranges["crop"].str.strip().str.lower())
    real = real[~real["label"].isin(extra_names)]        # idempotent re-runs
    real_crops = sorted(real["label"].unique())
    synth = generate_range_based(ranges, np.random.default_rng(SEED))
    data = pd.concat([real, synth], ignore_index=True)
    source = {c: "dataset" for c in real_crops}
    source.update({c: "range-based estimate" for c in sorted(extra_names)})
    return data, source, ranges, synth


def write_dataset_csv(ranges: pd.DataFrame, synth: pd.DataFrame):
    """Rewrite Crop_recommendation.csv: the original rows are kept byte-for-byte,
    and the generated rows are appended at the end (CRLF line endings, as before)."""
    extra = set(ranges["crop"].str.strip().str.lower())
    lines = DATA_CSV.read_text().splitlines()
    header, body = lines[0], [ln for ln in lines[1:] if ln.strip()]
    kept = [ln for ln in body if ln.rsplit(",", 1)[-1].strip().lower() not in extra]
    new = synth[FEATURES + ["label"]].to_csv(index=False, header=False, lineterminator="\n").splitlines()
    with open(DATA_CSV, "w", newline="") as f:
        f.write("\r\n".join([header] + kept + new) + "\r\n")


# ----------------------------------------------------------------------------
# model + evaluation
# ----------------------------------------------------------------------------
def make_rf():
    return RandomForestClassifier(
        n_estimators=N_TREES, max_features="sqrt", min_samples_leaf=MIN_LEAF,
        n_jobs=-1, random_state=SEED,
    )


def augment(X, y, sd, rng):
    Xs, ys = [X], [y]
    for _ in range(AUG_COPIES):
        Xs.append(X + rng.normal(0, 1, X.shape) * sd * AUG_NOISE)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def cross_validate(X, y, sd, label_source):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rng = np.random.default_rng(SEED)
    out = {k: {"clean": [], "top3": [], **{f"noise{n}": [] for n in NOISE_LEVELS}} for k in ("plain", "augmented")}
    per_crop_hits = {k: pd.Series(0.0, index=sorted(set(y))) for k in out}
    per_crop_n = pd.Series(0.0, index=sorted(set(y)))
    for tr, te in skf.split(X, y):
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        per_crop_n = per_crop_n.add(pd.Series(yte).value_counts(), fill_value=0)
        for name in out:
            Xfit, yfit = (Xtr, ytr) if name == "plain" else augment(Xtr, ytr, sd, rng)
            m = make_rf().fit(Xfit, yfit)
            pred = m.predict(Xte)
            out[name]["clean"].append((pred == yte).mean())
            proba = m.predict_proba(Xte)
            top3 = m.classes_[np.argsort(proba, axis=1)[:, -3:]]
            out[name]["top3"].append(np.mean([t in row for t, row in zip(yte, top3)]))
            for n in NOISE_LEVELS:
                Xn = Xte + rng.normal(0, 1, Xte.shape) * sd * n
                out[name][f"noise{n}"].append((m.predict(Xn) == yte).mean())
            hits = pd.Series(pred == yte, index=yte).groupby(level=0).sum()
            per_crop_hits[name] = per_crop_hits[name].add(hits, fill_value=0)
    summary = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in out.items()}
    per_crop = {k: (per_crop_hits[k] / per_crop_n) for k in out}
    return summary, per_crop


def main():
    data, source, ranges, synth = build_dataset()
    X = data[FEATURES].to_numpy(dtype=float)
    y = data["label"].to_numpy()
    sd = X.std(axis=0)

    print(f"Dataset: {len(data)} rows, {data['label'].nunique()} crops "
          f"({sum(v == 'dataset' for v in source.values())} from the dataset, "
          f"{sum(v != 'dataset' for v in source.values())} range-based)\n")

    summary, per_crop = cross_validate(X, y, sd, source)

    real_crops = [c for c, s in source.items() if s == "dataset"]
    new_crops = [c for c, s in source.items() if s != "dataset"]
    pc = per_crop["plain"]
    print(f"5-fold cross-validation (RandomForest, {N_TREES} trees)")
    for name, s in summary.items():
        print(f"  {name:9s} clean {s['clean']*100:6.2f}%  top-3 {s['top3']*100:6.2f}%  "
              + "  ".join(f"noise {n:.2f}sd {s[f'noise{n}']*100:5.1f}%" for n in NOISE_LEVELS))
    print(f"\n  Per-origin accuracy (plain): dataset crops {pc[real_crops].mean()*100:.2f}% | "
          f"range-based crops {pc[new_crops].mean()*100:.2f}%  <- the latter is optimistic (generated data)")
    print("  Weakest crops:", ", ".join(f"{c} {v*100:.0f}%" for c, v in pc.sort_values().head(5).items()))

    # choose augmented only if clean accuracy is not hurt and noisy accuracy improves
    p, a = summary["plain"], summary["augmented"]
    use_aug = (a["clean"] >= p["clean"] - 0.003) and (a[f"noise{NOISE_LEVELS[-1]}"] > p[f"noise{NOISE_LEVELS[-1]}"])
    chosen = "augmented" if use_aug else "plain"
    cv_clean = summary[chosen]["clean"]
    print(f"\nChosen training mode: {chosen}")

    Xfit, yfit = (augment(X, y, sd, np.random.default_rng(SEED)) if use_aug else (X, y))
    # Fit on a named DataFrame, like the original train_model.py, so the app's DataFrame input matches.
    model = make_rf().fit(pd.DataFrame(Xfit, columns=FEATURES), yfit)

    backed_up = False
    if MODEL_PKL.exists():
        shutil.copy(MODEL_PKL, HERE / "crop_model_backup_v1.pkl")
        backed_up = True
    bundle = {
        "model": model,
        "features": FEATURES,
        "model_name": "Random Forest" + (" (noise-augmented)" if use_aug else ""),
        "accuracy": float(cv_clean),                       # CV accuracy, 0-1 float (read by the app)
        "accuracy_type": "cross_validation_accuracy",
        "model_accuracy": float(cv_clean),
        "accuracy_note": "5-fold cross-validation on the combined dataset; "
                         "range-based crops are generated, so real-world accuracy is lower.",
        "cv_top3_accuracy": float(summary[chosen]["top3"]),
        "cv_noise_accuracy": {str(n): float(summary[chosen][f"noise{n}"]) for n in NOISE_LEVELS},
        "dataset_accuracy_only": float(per_crop[chosen][real_crops].mean()),
        "crop_data_source": source,
        "n_samples": int(len(data)),
        "n_crops": int(data["label"].nunique()),
        "sklearn_version": sklearn.__version__,
        "trained_on": datetime.now().strftime("%Y-%m-%d"),
    }
    joblib.dump(bundle, MODEL_PKL, compress=3)
    write_dataset_csv(ranges, synth)
    (HERE / "model_report.json").write_text(json.dumps(
        {k: v for k, v in bundle.items() if k != "model"}, indent=2))
    print(f"\nSaved {MODEL_PKL.name} ({bundle['n_crops']} crops), updated {DATA_CSV.name}, wrote model_report.json")
    if backed_up:
        print("Previous model kept as crop_model_backup_v1.pkl")


if __name__ == "__main__":
    main()
