"""Shared data loading and preprocessing for Python MLP implementations.

Mirrors the C data_loader.c behavior: same datasets, same normalization,
same train/test split logic for fair benchmark comparisons.
"""

import os
import struct
import sys
import numpy as np


def load_dataset(name, data_dir="data", num_samples=0):
    """Load a dataset by name, returning (X, y, num_classes).

    Args:
        name: One of 'generated', 'iris', 'wine-red', 'wine-white', 'breast-cancer'.
        data_dir: Path to data directory (relative to project root).
        num_samples: Number of samples for generated dataset (0 = default 1000).

    Returns:
        X: numpy array of shape (num_samples, num_features), dtype float32.
        y: numpy array of shape (num_samples,), dtype int32.
        num_classes: int.
    """
    if name == "generated":
        return load_generated_data(num_samples=num_samples)
    elif name == "iris":
        return load_iris_data(data_dir)
    elif name == "wine-red":
        return load_wine_quality_data(os.path.join(data_dir, "winequality-red.csv"))
    elif name == "wine-white":
        return load_wine_quality_data(os.path.join(data_dir, "winequality-white.csv"))
    elif name == "breast-cancer":
        return load_breast_cancer_data(data_dir)
    elif name == "mnist":
        return load_mnist_data(data_dir)
    else:
        print(f"Unknown dataset: {name}", file=sys.stderr)
        sys.exit(1)


def load_generated_data(num_samples=0):
    """Synthetic 2D circle classification matching C implementation.

    Points in [-1, 1]^2. Label 1 if inside circle of radius 0.5, else 0.
    num_samples <= 0 defaults to 1000.
    """
    np.random.seed(None)
    if num_samples <= 0:
        num_samples = 1000
    X = np.random.uniform(-1, 1, size=(num_samples, 2)).astype(np.float32)
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) < 0.25).astype(np.int32)
    return X, y, 2


def load_iris_data(data_dir="data"):
    """Load preprocessed Iris dataset.

    Format: comma-separated, 4 float features then float label per line.
    """
    filepath = os.path.join(data_dir, "iris_processed.txt")
    data = np.loadtxt(filepath, delimiter=",", dtype=np.float32)
    X = data[:, :4]
    y = np.round(data[:, 4]).astype(np.int32)
    return X, y, 3


def load_wine_quality_data(filepath):
    """Load UCI Wine Quality dataset.

    Semicolon-delimited CSV with header. 11 features, quality score 0-10 as label.
    """
    data = np.loadtxt(filepath, delimiter=";", skiprows=1, dtype=np.float32)
    X = data[:, :11]
    y = data[:, 11].astype(np.int32)
    return X, y, 11


def load_breast_cancer_data(data_dir="data"):
    """Load Wisconsin Breast Cancer Diagnostic dataset.

    Format: id, diagnosis (M/B), 30 float features. No header.
    M (malignant) = 1, B (benign) = 0.
    """
    filepath = os.path.join(data_dir, "wdbc.data")
    X_list = []
    y_list = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            # parts[0] = id, parts[1] = diagnosis, parts[2:] = 30 features
            diagnosis = 1 if parts[1] == "M" else 0
            features = [float(x) for x in parts[2:32]]
            X_list.append(features)
            y_list.append(diagnosis)
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y, 2


def load_mnist_data(data_dir="data", is_test=False):
    """Load MNIST IDX binary format. Returns (X, y, 10).

    Reads the IDX binary format directly -- no external dependencies needed.
    Pixel values normalized to [0, 1] by dividing by 255.
    """
    prefix = "t10k" if is_test else "train"
    img_path = os.path.join(data_dir, f"{prefix}-images-idx3-ubyte")
    lbl_path = os.path.join(data_dir, f"{prefix}-labels-idx1-ubyte")

    with open(img_path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        X = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8)
        X = X.reshape(n, rows * cols).astype(np.float32) / 255.0

    with open(lbl_path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        y = np.frombuffer(f.read(n), dtype=np.uint8).astype(np.int32)

    return X, y, 10


def normalize_features(X):
    """Z-score normalization per feature, matching C implementation.

    x' = (x - mean) / sqrt(var + epsilon), epsilon = 1e-8.
    """
    mean = X.mean(axis=0)
    std = np.sqrt(((X - mean) ** 2).mean(axis=0) + 1e-8)
    return (X - mean) / std


def shuffle_and_split(X, y, train_frac=0.8):
    """Randomly shuffle then split into train/test sets.

    Returns:
        X_train, y_train, X_test, y_test
    """
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    split = int(len(X) * train_frac)
    return X[:split], y[:split], X[split:], y[split:]
