"""
Stutter Classifier Training Script
=====================================
Trains a lightweight sklearn RandomForest classifier on MFCC features
extracted from the SEP-28k stuttering event dataset.

Usage:
    python utils/train_classifier.py --data_dir /path/to/sep28k_wavs

SEP-28k Dataset:
    https://github.com/apple/ml-stuttering-events-dataset
    Paper: Lea et al. 2021, "SEP-28k: A Dataset for Stuttering Event Detection"

If you don't have SEP-28k, the system falls back to heuristic rules automatically.
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

LABEL_MAP = {
    "Fluent": "fluent",
    "Prolongation": "stutter",
    "Block": "stutter",
    "SoundRep": "repetition",
    "WordRep": "repetition",
    "Interjection": "filler",
    "NoSpeech": "long_pause",
}

OUTPUT_LABELS = ["fluent", "stutter", "filler", "long_pause", "repetition"]


def extract_features_from_wav(wav_path: Path) -> np.ndarray:
    """Extract 44-dimensional feature vector from WAV file."""
    import librosa

    audio, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    if len(audio) < 256:
        return None

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))

    try:
        pitches, mags = librosa.piptrack(y=audio, sr=sr, fmin=75, fmax=400)
        pitch_vals = []
        for t in range(pitches.shape[1]):
            idx = mags[:, t].argmax()
            p = pitches[idx, t]
            if p > 0:
                pitch_vals.append(p)
        pitch_mean = float(np.mean(pitch_vals)) if pitch_vals else 0.0
        pitch_std = float(np.std(pitch_vals)) if pitch_vals else 0.0
    except Exception:
        pitch_mean, pitch_std = 0.0, 0.0

    return np.concatenate([
        np.mean(mfcc, axis=1),        # 13
        np.std(mfcc, axis=1),         # 13
        np.mean(mfcc_delta, axis=1),  # 13
        [zcr, rms, centroid, pitch_mean, pitch_std],  # 5
    ])  # Total: 44


def train_from_sep28k(data_dir: str, output_path: str = None):
    data_dir = Path(data_dir)
    output_path = output_path or str(Path(__file__).parent.parent / "models" / "stutter_classifier.pkl")

    logger.info(f"Loading SEP-28k data from {data_dir}")

    # Try to find the labels CSV
    labels_csv = data_dir / "SEP-28k-Extended.csv"
    if not labels_csv.exists():
        labels_csv = next(data_dir.glob("*.csv"), None)

    if not labels_csv:
        logger.error("No CSV labels file found in data directory")
        return

    import pandas as pd
    df = pd.read_csv(labels_csv)
    logger.info(f"Loaded {len(df)} entries from labels CSV")

    X, y = [], []
    for _, row in df.iterrows():
        # Try to locate the corresponding audio file
        wav_candidates = list(data_dir.rglob(f"*{row.get('Show', '')}*{row.get('EpId', '')}*"))
        if not wav_candidates:
            continue

        features = extract_features_from_wav(wav_candidates[0])
        if features is None:
            continue

        # Map the annotation to our simplified labels
        label = "fluent"
        for col in ["Prolongation", "Block", "SoundRep", "WordRep", "Interjection", "NoSpeech"]:
            if col in df.columns and row.get(col, 0) > 0:
                label = LABEL_MAP.get(col, "fluent")
                break

        X.append(features)
        y.append(label)

    if len(X) < 50:
        logger.error(f"Only {len(X)} valid samples found — not enough to train")
        return

    X = np.array(X)
    y = np.array(y)
    logger.info(f"Dataset: {len(X)} samples, {len(set(y))} classes")
    logger.info(f"Class distribution: { {c: (y == c).sum() for c in set(y)} }")

    # Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Evaluate
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    logger.info(f"Cross-val accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(classification_report(y_test, clf.predict(X_test)))

    # Save
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "wb") as f:
        pickle.dump(clf, f)
    logger.info(f"Saved classifier to {output_path}")


def generate_synthetic_dataset(n_samples: int = 2000, output_path: str = None):
    """
    Generate a synthetic training dataset using acoustic heuristics.
    Useful for testing the pipeline without the SEP-28k dataset.
    """
    output_path = output_path or str(Path(__file__).parent.parent / "models" / "stutter_classifier.pkl")

    logger.info("Generating synthetic training data...")
    rng = np.random.default_rng(42)

    def _gen_sample(label: str) -> np.ndarray:
        base = rng.normal(0, 1, 44)
        if label == "fluent":
            base[39] = rng.uniform(0.05, 0.15)  # rms: moderate
            base[38] = rng.uniform(0.02, 0.08)  # zcr: low
            base[41] = rng.uniform(100, 250)     # pitch_mean: normal
        elif label == "stutter":
            base[39] = rng.uniform(0.01, 0.08)  # rms: variable
            base[38] = rng.uniform(0.1, 0.25)   # zcr: high
            base[:13] = rng.normal(0, 5, 13)    # erratic MFCCs
        elif label == "long_pause":
            base[39] = rng.uniform(0.0, 0.01)   # rms: near zero
            base[38] = rng.uniform(0.0, 0.03)
        elif label == "filler":
            base[39] = rng.uniform(0.03, 0.10)
            base[41] = rng.uniform(150, 300)
        elif label == "repetition":
            base[39] = rng.uniform(0.04, 0.12)
            base[43] = rng.uniform(50, 100)     # high pitch_std
        return base

    labels = ["fluent", "stutter", "long_pause", "filler", "repetition"]
    weights = [0.55, 0.20, 0.10, 0.10, 0.05]

    X, y = [], []
    for _ in range(n_samples):
        label = rng.choice(labels, p=weights)
        X.append(_gen_sample(label))
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=80, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    acc = (clf.predict(X_test) == y_test).mean()
    logger.info(f"Synthetic classifier accuracy: {acc:.3f} (expected ~0.75 on synthetic data)")

    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "wb") as f:
        pickle.dump(clf, f)
    logger.info(f"Saved synthetic classifier to {output_path}")
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to SEP-28k dataset directory")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic training data (no real dataset needed)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.synthetic or args.data_dir is None:
        generate_synthetic_dataset(output_path=args.output)
    else:
        train_from_sep28k(args.data_dir, args.output)
