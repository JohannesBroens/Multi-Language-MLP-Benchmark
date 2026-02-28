#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Determine the project root directory
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "NOTE: For full pipeline with config support, use: python src/scripts/pipeline.py all --model mlp"
echo ""
echo "Starting the legacy pipeline..."

# Step 1: Download datasets
echo "Checking for datasets..."
chmod +x "$SCRIPT_DIR/download_datasets.sh"
"$SCRIPT_DIR/download_datasets.sh"

# Step 2: Preprocess datasets
echo "Checking for preprocessed data..."
python3 "$SCRIPT_DIR/preprocess_iris.py"

# Step 3: Build the project (CPU by default, CUDA if available)
echo "Building the project..."

# CPU build
mkdir -p "$PROJECT_ROOT/src/c/build_cpu"
cd "$PROJECT_ROOT/src/c/build_cpu"
cmake .. -DUSE_CUDA=OFF
make

# CUDA build (attempt, skip if no CUDA)
if command -v nvcc &> /dev/null; then
    echo "CUDA detected, building CUDA version..."
    mkdir -p "$PROJECT_ROOT/src/c/build_cuda"
    cd "$PROJECT_ROOT/src/c/build_cuda"
    cmake .. -DUSE_CUDA=ON
    make
else
    echo "CUDA not found, skipping CUDA build."
fi

# Step 4: Run the program
echo "Running the program..."

cd "$PROJECT_ROOT"

if [ $# -eq 1 ]; then
    DATASET=$1
    echo "Running on dataset: $DATASET"
    ./src/c/build_cpu/main --dataset "$DATASET"
else
    DATASETS=("generated" "iris" "wine-red" "wine-white" "breast-cancer")
    for DATASET in "${DATASETS[@]}"; do
        echo "--------------------------------------------"
        echo "Running on dataset: $DATASET"
        ./src/c/build_cpu/main --dataset "$DATASET"
    done
fi

echo "Pipeline completed."
