import os
import torch
import pickle
import statistics
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F


def load_file(filepath):
    """
    Helper function to load a single .npy file.
    Returns the loaded array if successful; otherwise, returns None.
    """
    if os.path.isfile(filepath):
        try:
            arr = np.load(filepath)
            return arr
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    else:
        print(f"File not found: {filepath}")
        return None

    
def load_embeddings(path, langs, model_configs, max_workers=64):
    """
    Load embeddings for each model and language in parallel while preserving the order.
    
    Parameters:
        path (str): Base directory or prefix for the embedding files.
        langs (list): List of language codes.
        model_configs (list): List of dictionaries, each with model information (e.g. 'name', 'prefix').
        max_workers (int): Maximum number of worker threads to use.
    
    Returns:
        embeddings (dict): Maps each model name to a concatenated numpy array of embeddings.
        langs_exists (dict): Maps each model name to the list of languages for which embeddings were successfully loaded.
    """
    embeddings = {}
    langs_exists = {}
    # Outer progress bar: iterate over models.
    for model in tqdm(model_configs, desc="Loading models"):
        model_name = model['name']
        model_prefix = model['prefix']
        print(f"\nLoading embeddings for {model_name}")
        
        model_embs = []
        lang_list = []
        # Submit one task per language while preserving order.
        future_map = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for lang in langs:
                filepath = f"{path}/{model_prefix}_{lang}.npy"
                future_map[lang] = executor.submit(load_file, filepath)
            # Inner progress bar: iterate over languages for the current model.
            for lang in tqdm(langs, desc=f"Loading files for {model_name}", leave=False):
                result = future_map[lang].result()
                if result is not None:
                    model_embs.append(result)
                    lang_list.append(lang)
                else:
                    print(f"Error loading {lang} for {model_name}")
        if model_embs:
            embeddings[model_name] = np.concatenate(model_embs, axis=0)
            langs_exists[model_name] = lang_list
        else:
            embeddings[model_name] = None
            langs_exists[model_name] = []
    return embeddings, langs_exists


def get_neighbors_indices(X, neighbor_k=10, metric='euclidean'):
    """
    Compute pairwise distances for X on the GPU and return the indices and distances 
    of the k-fixed nearest neighbors for each point using the specified metric.
    
    Parameters:
        X (torch.Tensor): Input tensor of shape (N, features).
        neighbor_k (int): Number of neighbors to retrieve (excluding the point itself).
        metric (str): Distance metric to use ('euclidean' or 'cosine').
        
    Returns:
        neighbor_indices (np.ndarray): An array of shape (N, neighbor_k) with indices.
        neighbor_distances (np.ndarray): An array of shape (N, neighbor_k) with distances.
        R: Always None (kept for interface compatibility).
    """
    with torch.no_grad():
        if metric == 'euclidean':
            dists = torch.cdist(X, X, p=2)
        elif metric == 'cosine':
            # Normalize X along the feature dimension
            X_norm = F.normalize(X, p=2, dim=1)
            # Compute cosine similarity matrix and then convert it to cosine distance
            similarity = torch.matmul(X_norm, X_norm.T)
            dists = 1 - similarity
        else:
            raise ValueError("Unsupported distance metric: {}. Use 'euclidean' or 'cosine'.".format(metric))
        
        # Retrieve k+1 indices because the closest is the point itself.
        _, idx_temp = torch.topk(-dists, k=neighbor_k+1, dim=1)
        
    # Exclude the self-index (first column) and move to CPU.
    idx_temp = idx_temp.cpu().numpy()[:, 1:]
    dists_np = dists.cpu().numpy()
    # Use vectorized indexing to extract the distances for the selected neighbor indices.
    neighbor_distances = np.take_along_axis(dists_np, idx_temp, axis=1)
    R = None
    return idx_temp, neighbor_distances, R


def compute_bridge_reach(emb_by_lang, neighbor_k=10, neighbor_threshold=3, metric='euclidean', device='cuda'):
    """
    Compute Bridge and Reachability metrics.
    
    For each point (from the concatenated embeddings) the k-fixed nearest neighbors are computed.
    A point is considered "bridging" if the set of neighbor languages (excluding its own)
    has at least 'neighbor_threshold' distinct languages.
    
    Returns:
        bridge (dict): Mapping language -> raw bridge score (fraction).
        reach (dict): Mapping language -> raw reachability (distinct neighboring languages count).
    """
    X, labels = [], []
    for lang, arr in emb_by_lang.items():
        # Ensure the array is a GPU tensor.
        if isinstance(arr, np.ndarray):
            tensor = torch.tensor(arr, device=device, dtype=torch.float32)
        else:
            tensor = arr.to(device, dtype=torch.float32)
        X.append(tensor)
        labels += [lang] * tensor.shape[0]
    if len(X) == 0:
        return {}, {}
    with torch.no_grad():
        X = torch.cat(X, dim=0)
    labels = np.array(labels)
    neighbor_indices, neighbor_distances, _ = get_neighbors_indices(X, neighbor_k=neighbor_k, metric=metric)
    
    bridge = {lang: 0 for lang in emb_by_lang}  # Count bridging points.
    reach = {lang: set() for lang in emb_by_lang}  # Distinct neighboring languages.
    lang_counts = {lang: 0 for lang in emb_by_lang}
    
    for lab in labels:
        lang_counts[lab] += 1
    
    for i in range(X.shape[0]):
        lang_i = labels[i]
        indices = neighbor_indices[i]
        neighbor_labels = set(labels[indices]) - {lang_i}
        if len(neighbor_labels) >= neighbor_threshold:
            bridge[lang_i] += 1
        reach[lang_i].update(neighbor_labels)
    
    # Normalize and convert reach sets to counts.
    for lang in bridge:
        bridge[lang] = bridge[lang] / lang_counts[lang] if lang_counts[lang] > 0 else 0.0
        reach[lang] = len(reach[lang])
    
    return bridge, reach


def derive_metrics(embeddings, langs_exists, model_configs, output_csv_path, langs_analyzed=None,
                   bridge_ratio=0.7, reach_ratio=0.3, selected_layer=10, neighbor_k=10, 
                   neighbor_threshold=3, threshold=0.2, method='fixed', metric='euclidean', device='cuda'):
    """
    Compute metrics (bridge, reachability, combined score, average distinct languages) for each language,
    and output the results to a CSV file.
    
    **Note:** The variable `langs_analyzed` must be defined externally.
    """
    model = model_configs[0]['name']
    all_emb = embeddings[model][:, selected_layer, :]
    langs_list = langs_exists[model]
    langs_list = [lang for lang in langs_list if lang in langs_analyzed]
    if not langs_list:
        print("None of the loaded languages are in the target analysis list.")
        return
    lang_emb = {
        lang: torch.from_numpy(all_emb[i * 997:(i + 1) * 997]).to(device=device, dtype=torch.float32)
        for i, lang in enumerate(langs_list)
    }
    total_languages = len(lang_emb)
    
    bridge, reach = compute_bridge_reach(lang_emb, neighbor_k=neighbor_k, neighbor_threshold=neighbor_threshold, metric=metric)
    norm_bridges = {lang: bridge[lang] for lang in bridge}  # Already normalized.
    norm_reaches = {lang: reach[lang] / (total_languages - 1) for lang in reach}

    languages = list(lang_emb.keys())
    norm_bridges = [norm_bridges[lang] for lang in languages]
    norm_reaches = [norm_reaches[lang] for lang in languages]
    ilo_scores = [statistics.harmonic_mean([bridge, reach]) for bridge, reach in zip(norm_bridges, norm_reaches)]
    
    data = {
        'language': languages,
        'raw_bridge': [bridge[lang] for lang in languages],
        'normalized_bridge': norm_bridges,
        'raw_reach': [reach[lang] for lang in languages],
        'normalized_reach': norm_reaches,
        'ilo_scores': ilo_scores,
    }
    df = pd.DataFrame(data)
    df['harm_bridge_reach'] = df.apply(lambda x: statistics.harmonic_mean([x['normalized_bridge'], x['normalized_reach']]), axis=1)
    return df