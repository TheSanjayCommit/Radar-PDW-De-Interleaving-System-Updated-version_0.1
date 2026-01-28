import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans

# Try importing HDBSCAN
try:
    from sklearn.cluster import HDBSCAN
    HAS_HDBSCAN = True
    HDBSCAN_LIB = "sklearn"
except ImportError:
    try:
        import hdbscan
        HAS_HDBSCAN = True
        HDBSCAN_LIB = "hdbscan"
    except ImportError:
        HAS_HDBSCAN = False
        HDBSCAN_LIB = None

def auto_tune_dbscan(df, features, known_emitters):
    """
    Iteratively finds the best Epsilon for DBSCAN to match known_emitters.
    """
    X = StandardScaler().fit_transform(df[features].values)
    best_err = float("inf")
    best_eps = 0.5
    best_ms = 5

    for eps in np.arange(0.1, 3.0, 0.1):
        db = DBSCAN(eps=eps, min_samples=5)
        labels = db.fit_predict(X)
        clusters = len(set(labels)) - (1 if -1 in labels else 0)
        err = abs(clusters - known_emitters)
        if err < best_err:
            best_err = err
            best_eps = eps
            best_ms = 5
        if err == 0:
            break
            
    return {"eps": float(best_eps), "min_samples": best_ms}


def run_clustering(algorithm, df, features, params, custom_tols=None):
    """
    Executes the selected clustering algorithm and returns labels.
    """
    labels = []
    
    if algorithm == "DBSCAN":
        # Apply Custom Scaling based on Tolerances
        X_custom = df[features].copy()
        
        for col in features:
            if custom_tols and col in custom_tols:
                 X_custom[col] = X_custom[col] / custom_tols[col]
        
        X_final = X_custom.fillna(0).values
        labels = DBSCAN(**params).fit_predict(X_final)

    elif algorithm == "HDBSCAN":
        X = StandardScaler().fit_transform(df[features].values)
        if HDBSCAN_LIB == "sklearn":
            labels = HDBSCAN(**params).fit_predict(X)
        else:
            labels = hdbscan.HDBSCAN(**params).fit_predict(X)

    elif algorithm == "K-Means":
        X = StandardScaler().fit_transform(df[features].values)
        labels = KMeans(**params, random_state=42, n_init=10).fit_predict(X)

    return labels


def process_results(labels, known_emitters):
    """
    Maps raw labels to 1-indexed Emitter IDs and generates summary stats.
    """
    unique = sorted(set(labels))
    label_map = {l: i + 1 for i, l in enumerate(unique) if l != -1}
    label_map[-1] = 0

    mapped_labels = [label_map[l] for l in labels]
    
    summary = {
        "clusters": len(set(mapped_labels)) - (1 if 0 in mapped_labels else 0),
        "noise": mapped_labels.count(0),
        "expected": known_emitters
    }
    
    return mapped_labels, summary
