#!/usr/bin/env python3
"""
Calculate Interlinguality (ILO) from hidden-state representations.

This script:
1) Extracts per-layer sequence embeddings for multiple languages from the FLORES dataset.
2) Loads those embeddings and computes interlinguality metrics per model layer.
"""

import argparse
import glob
import os
from pathlib import Path
import pickle
from typing import List, Dict, Any

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.ilo import load_embeddings, derive_metrics


# Languages to process
LANGS: List[str] = [
    "swe_Latn", "nld_Latn", "bjn_Latn", "jav_Latn", "tgl_Latn",
    "ind_Latn", "rus_Cyrl", "ces_Latn", "dan_Latn", "eng_Latn",
    "ukr_Cyrl", "tel_Telu", "srp_Cyrl", "fra_Latn", "tha_Thai",
    "ban_Latn", "zho_Hans", "deu_Latn", "sun_Latn", "hin_Deva",
    "pol_Latn", "slv_Latn", "yue_Hant", "sin_Sinh", "spa_Latn",
    "urd_Arab", "gle_Latn", "swh_Latn", "ben_Beng", "jpn_Jpan",
    "min_Latn",
]

# Filename pieces
FLORES_DATASET_NAME = "flores_200"
FLORES_HF_REPO = "facebook/flores"
EMBED_SAVE_DIR = Path(f"save/embeddings/{FLORES_DATASET_NAME}")
METRICS_SAVE_DIR = Path("save/metrics")
ANALYSIS_TYPE = "tsne_emb_"
DATASET_TAG = "flores_"


def _sanitize_model_name(model_name: str) -> str:
    """Sanitize a HF model name for use in filenames."""
    return model_name.replace("/", "_").replace(".", "").replace("-", "_")


def extract_embeddings(model_name: str, langs: List[str]) -> None:
    """
    Extract and save mean token embeddings per layer, per example, for each language.
    Each saved .npy has shape: (num_examples, num_layers+1, hidden_dim)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Step 1/2] Getting embeddings from {FLORES_DATASET_NAME} in {len(langs)} languages")
    print(f"Model: {model_name} | Device: {device}")

    # Load model & tokenizer once and reuse for all languages
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
    model.eval()

    model_filename = _sanitize_model_name(model_name)

    for lang in tqdm(langs, desc="Languages"):
        save_path = EMBED_SAVE_DIR / f"{ANALYSIS_TYPE}{DATASET_TAG}{model_filename}_{lang}.npy"
        if save_path.is_file():
            print(f"✓ Found cached embeddings for {lang}: {save_path}")
            continue

        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"• Creating embeddings for {lang} -> {save_path}")

        # Load FLORES split for the language
        ds = load_dataset(FLORES_HF_REPO, lang, trust_remote_code=True)["dev"]

        all_embs = []
        for example in tqdm(ds, desc=f"Examples ({lang})", leave=False):
            input_text = example["sentence"]

            # Tokenize (truncate to keep memory predictable)
            tokens = tokenizer(
                input_text,
                return_tensors="pt"
            )
            input_ids = tokens["input_ids"].to(device)

            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)

            # hidden_states is a tuple: (layers+1) x [B, T, H]
            # Stack -> [L+1, B, T, H] ; then mean over batch (B) and tokens (T) -> [L+1, H]
            hidden_states = torch.stack(outputs.hidden_states, dim=0)  # [L+1, B, T, H]
            layer_token_mean = hidden_states.mean(dim=1).mean(dim=1)   # -> [L+1, H]
            all_embs.append(layer_token_mean.cpu())

        # [N, L+1, H] float32
        all_embs_tensor = torch.stack(all_embs, dim=0).float().numpy()
        np.save(save_path.as_posix(), all_embs_tensor)
        print(f"Saved {all_embs_tensor.shape} to {save_path}")


def discover_model_configs() -> (List[Dict[str, Any]], List[str]):
    """
    Scan the embeddings directory to build model config entries and list languages found.
    Returns:
        model_configs: [{'name': <sanitized_model_name>, 'prefix': 'tsne_emb_flores_<sanitized_model_name>'}, ...]
        langs_from_files: unique language IDs discovered from filenames
    """
    path_clue = (EMBED_SAVE_DIR / f"{ANALYSIS_TYPE}{DATASET_TAG}").as_posix()
    model_names, langs_from_files = [], []

    for fp in glob.glob(path_clue + "*"):
        # fp ends with "... tsne_emb_flores_<model>_<lang>.npy"
        tail = fp[len(path_clue):]
        parts = tail.split("_")
        model_names.append("_".join(parts[:-2]))  # everything except last two lang parts
        lang_id = "_".join(parts[-2:])[:-4]       # last two parts minus '.npy'
        langs_from_files.append(lang_id)

    model_names = sorted(list(set(model_names)), key=lambda x: x.lower())
    model_configs = [{"name": m, "prefix": f"{ANALYSIS_TYPE}{DATASET_TAG}{m}"} for m in model_names]

    print("Languages (from files):", sorted(list(set(langs_from_files))))
    print("Available model configs:", model_configs)
    print(f"Selected {len(model_configs)} model configs")
    return model_configs, sorted(list(set(langs_from_files)))


def compute_ilo(
    neighbor_k: int,
    neighbor_threshold: int,
    metric: str,
    langs_analyzed: List[str],
    metrics_filename: Path,
) -> None:
    """
    Compute interlinguality metrics per layer for each discovered model config.
    Saves a pickle of DataFrames keyed by model then layer.
    """
    # Build configs by scanning what's on disk
    model_configs, _ = discover_model_configs()

    if metrics_filename.is_file():
        print(f"✓ Metrics already exist at: {metrics_filename}")
        return

    # Load embeddings for all discovered model configs
    embeddings_path = EMBED_SAVE_DIR.as_posix()
    embeddings, langs_exists = load_embeddings(embeddings_path, langs_analyzed, model_configs)

    print(f"[Step 2/2] Deriving ILO (k={neighbor_k}, t={neighbor_threshold}, metric={metric})")
    per_layer_dfs: Dict[str, Dict[int, Any]] = {}

    for cfg in tqdm(model_configs, desc="Models"):
        model_key = cfg["name"]
        per_layer_dfs[model_key] = {}

        # Expect embeddings[model_key] -> [N, L, H]
        _, n_layer, _ = embeddings[model_key].shape
        print(f"Computing layers for model: {model_key} (n_layers={n_layer})")

        for layer in tqdm(range(n_layer), desc=f"Layers ({model_key})", leave=False):
            output_csv_path = "temp_interlinguality.csv"
            df = derive_metrics(
                embeddings,
                langs_exists,
                [cfg],
                output_csv_path,
                langs_analyzed=langs_analyzed,
                selected_layer=layer,
                neighbor_k=neighbor_k,
                neighbor_threshold=neighbor_threshold,
                bridge_ratio=0.5,
                reach_ratio=0.5,
                method="fixed",
                metric=metric,
            )
            per_layer_dfs[model_key][layer] = df

    metrics_filename.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_filename, "wb") as f:
        pickle.dump(per_layer_dfs, f)

    print(f"✓ Saved ILO metrics to {metrics_filename}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate Interlinguality")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B",
        help="Hugging Face model id to use for embeddings.",
    )
    parser.add_argument(
        "--neighbor_k",
        type=int,
        default=5,
        help="Number of neighbors to consider (k).",
    )
    parser.add_argument(
        "--neighbor_threshold",
        type=int,
        default=3,
        help="Threshold for neighbor overlap (t).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        help="Distance metric to use (e.g., 'cosine', 'euclidean').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1) Extract & cache embeddings (only runs for languages missing on disk)
    extract_embeddings(model_name=args.model_name, langs=LANGS)

    # 2) Compute ILO
    metrics_path = METRICS_SAVE_DIR / f"interlinguality_k{args.neighbor_k}_t{args.neighbor_threshold}_{args.metric}.pkl"
    compute_ilo(
        neighbor_k=args.neighbor_k,
        neighbor_threshold=args.neighbor_threshold,
        metric=args.metric,
        langs_analyzed=LANGS,
        metrics_filename=metrics_path,
    )


if __name__ == "__main__":
    main()
