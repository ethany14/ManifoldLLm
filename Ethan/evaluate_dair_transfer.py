"""Transfer frozen EmoBank encoders to dair-ai/emotion classification.

Qwen and the AE encoders never see dair labels. A linear classifier is fitted on
the dair training split; C is selected on dev and reported once on test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from compare_geometry_targets import train_model
from compare_representation_routes import TARGETS, seed_everything
from run_ablation_study import encode


def probe(name, z, labels, train, dev, test):
    scaler = StandardScaler().fit(z[train])
    z = scaler.transform(z)
    candidates = []
    for c in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=c, max_iter=1000, class_weight=None)
        clf.fit(z[train], labels[train])
        score = f1_score(labels[dev], clf.predict(z[dev]), average="macro")
        candidates.append((score, c))
    best_dev, best_c = max(candidates, key=lambda item: item[0])
    clf = LogisticRegression(C=best_c, max_iter=1000, class_weight=None)
    clf.fit(z[np.concatenate([train, dev])], labels[np.concatenate([train, dev])])
    prediction = clf.predict(z[test])
    return {
        "method": name,
        "dimensions": int(z.shape[1]),
        "selected_c": best_c,
        "dev_macro_f1": float(best_dev),
        "test_macro_f1": float(f1_score(labels[test], prediction, average="macro")),
        "test_accuracy": float(accuracy_score(labels[test], prediction)),
        "test_balanced_accuracy": float(balanced_accuracy_score(labels[test], prediction)),
        "per_class_f1": {str(i): float(f1_score(labels[test], prediction, labels=[i], average="macro")) for i in range(6)},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emobank-csv", required=True)
    p.add_argument("--emobank-embeddings", required=True)
    p.add_argument("--dair-csv", required=True)
    p.add_argument("--dair-embeddings", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dimensions", type=int, default=16)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--reconstruction-weight", type=float, default=0.1)
    p.add_argument("--geometry-weight", type=float, default=0.1)
    a = p.parse_args()
    seed_everything(a.seed)
    emo = pd.read_csv(a.emobank_csv)
    dair = pd.read_csv(a.dair_csv)
    emo_raw = np.load(a.emobank_embeddings).astype(np.float32)
    dair_raw = np.load(a.dair_embeddings).astype(np.float32)
    if len(emo) != len(emo_raw) or len(dair) != len(dair_raw):
        raise ValueError("CSV and embedding row counts do not match")
    emo_train = np.flatnonzero(emo.split.eq("train"))
    emo_dev = np.flatnonzero(emo.split.eq("dev"))
    dair_train = np.flatnonzero(dair.split.eq("train"))
    dair_dev = np.flatnonzero(dair.split.eq("dev"))
    dair_test = np.flatnonzero(dair.split.eq("test"))
    labels = dair.label.to_numpy(np.int64)
    input_scaler = StandardScaler().fit(emo_raw[emo_train])
    emo_x = input_scaler.transform(emo_raw).astype(np.float32)
    dair_x = input_scaler.transform(dair_raw).astype(np.float32)
    emo_y_raw = emo[TARGETS].to_numpy(np.float32)
    emo_y = StandardScaler().fit(emo_y_raw[emo_train]).transform(emo_y_raw).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [probe("Raw_Qwen", dair_x, labels, dair_train, dair_dev, dair_test)]
    pca = PCA(n_components=a.dimensions, random_state=a.seed).fit(emo_x[emo_train])
    rows.append(probe("EmoBank_PCA", pca.transform(dair_x), labels, dair_train, dair_dev, dair_test))
    for target in ["none", "affective"]:
        seed_everything(a.seed)
        print(f"Training EmoBank encoder: {target}", flush=True)
        model = train_model(
            emo_x, emo_y, emo_train, emo_dev, a.dimensions, a.hidden_dim,
            a.epochs, a.batch_size, device, a.reconstruction_weight,
            a.geometry_weight, target, 0.5,
        )
        z = encode(model, dair_x, device, a.batch_size)
        rows.append(probe(f"AE_VAD_{target}", z, labels, dair_train, dair_dev, dair_test))
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    flat = pd.DataFrame([{k:v for k,v in row.items() if k != "per_class_f1"} for row in rows])
    flat.to_csv(output / "dair_transfer_results.csv", index=False)
    (output / "dair_transfer_summary.json").write_text(json.dumps({"settings":vars(a),"results":rows},indent=2),encoding="utf-8")
    print(flat.to_string(index=False))

if __name__ == "__main__":
    main()
