# Bayesian Network Generator with Graph Neural Networks

A Python project for generating synthetic Bayesian networks and training graph neural network models to learn their inference patterns.

---

## 📋 Overview

This project:
- **Generates** synthetic Bayesian networks with configurable sizes and structures
- **Preprocesses** networks into graph data suitable for machine learning
- **Trains** three types of Graph Neural Networks (GAT, GCN, GraphSAGE) to predict inference results
- **Benchmarks** models across different configurations
- **Analyzes** performance and statistical properties

---

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

1. **Generate Bayesian Networks:**
   ```bash
   python bayesian_network_generator.py
   ```

2. **Preprocess Data:**
   ```bash
   python data_preprocessor.py
   ```

3. **Train a Model:**
   ```bash
   python train_gat.py    # Graph Attention Network
   python train_gcn.py    # Graph Convolutional Network
   python train_graphsage.py  # GraphSAGE
   ```

---

## 📁 Project Structure

```
├── Core Training Scripts
│   ├── train_gat.py              # Train GAT model
│   ├── train_gcn.py              # Train GCN model
│   ├── train_graphsage.py        # Train GraphSAGE model
│   ├── gat_model.py              # GAT architecture
│   ├── gcn_model.py              # GCN architecture
│   └── graphsage_model.py        # GraphSAGE architecture
│
├── Data Generation & Processing
│   ├── bayesian_network_generator.py    # Generate synthetic BN
│   ├── bayesian_network_builder.py      # Build BN structures
│   ├── data_preprocessor.py             # Prepare data for training
│   ├── config_loader.py                 # Config utilities
│   └── exporter.py                      # Save/load utilities
│
├── Analysis & Evaluation
│   ├── benchmarking.py                  # Compare model performance
│   ├── outlier_analysis.py              # Detect outliers
│   ├── temperature_scaling.py           # Calibrate predictions
│   ├── statistical_analysis.py          # Statistical tests
│   └── compare_models.py                # Model comparison
│
├── Configuration
│   ├── config.yaml                      # Main settings file
│   └── sweep.yaml                       # Hyperparameter sweep config
│
├── Datasets & Results
│   ├── datasets/                        # Processed data splits
│   ├── data_processing/                 # Intermediate data
│   ├── global_datasets/                 # Global statistics
│   ├── benchmark_results/               # Benchmark outputs
│   ├── training_results_*/              # Model checkpoints & metrics
│   └── comparison_results/              # Comparison analysis
│
└── Utilities
    ├── graph_visualization.py           # Visualize graphs
    ├── test_small_networks.py           # Debug tests
    └── cleanup.py                       # Clean temporary files
```

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

### Graph Generation
- `num_graphs`: Number of Bayesian networks to generate (default: 40,000)
- `min_depth` / `max_depth`: Tree depth constraints
- `min_nodes` / `max_nodes`: Node count range
- `max_children`: Max children per node

### Training
- `learning_rate`: Adam optimizer learning rate (default: 0.0003)
- `batch_size`: Training batch size (default: 128)
- `epochs`: Max training epochs (default: 100)
- `patience`: Early stopping patience (default: 15)
- `dropout`: Dropout rate (default: 0.1)
- `heads`: Attention heads for GAT (default: 2)
- `hidden_channels`: Hidden layer size (default: 128)

### Inference
- `mode`: "distribution", "root_probability", or "regression"
- `mask_strategy`: "root_only", "both", "evidence_only", "none"
- `use_kfold`: Enable k-fold cross-validation (default: true)
- `k_folds`: Number of folds (default: 5)
- `use_temperature_scaling`: Calibrate predictions (default: true)

---

## 🔬 Key Features

### Multiple Model Architectures
- **GAT** (Graph Attention Network) - Attention-based learning
- **GCN** (Graph Convolutional Network) - Spectral convolutions
- **GraphSAGE** - Sampling and aggregating

### Advanced Techniques
- **Temperature Scaling** - Post-hoc probability calibration
- **K-Fold Cross-Validation** - Robust evaluation
- **Outlier Analysis** - Detect problematic predictions
- **Early Stopping** - Prevent overfitting

### Comprehensive Analysis
- Model comparison tools
- Distribution analysis
- Structural outlier detection
- Benchmark reports

---

## 📊 Expected Outputs

After training, check:
- `training_results_gat/models/` - Saved model weights
- `training_results_gat/plots/` - Loss/accuracy curves
- `training_results_gat/metrics/` - Evaluation metrics
- `benchmark_results/` - Comparative analysis
- Console logs - Training progress & validation scores

---

## 🛠️ Common Tasks

### Run Hyperparameter Sweep
```bash
python train_gat.py  # Configure sweep.yaml first
```

### Benchmark All Models
```bash
python benchmarking.py
```

### Analyze Model Outputs
```bash
python outlier_analysis.py
python statistical_analysis.py
```

### Compare Distributions
```bash
python compare_distributions.py
```

---

## 📦 Dependencies

Key packages:
- **PyTorch** + **PyG** (Geometric) - Graph neural networks
- **PyYAML** - Configuration management
- **scikit-learn** - ML utilities & metrics
- **bnlearn** - Bayesian network operations
- **networkx** - Graph algorithms
- **pandas, numpy** - Data manipulation
- **matplotlib, seaborn** - Visualization
- **wandb** - Experiment tracking

See `requirements.txt` for full list.

---

## 🔍 Troubleshooting

**Out of Memory:**
- Reduce `batch_size` in config.yaml
- Decrease `num_graphs`
- Use a machine with more GPU memory

**Training Stalls:**
- Increase `learning_rate` slightly
- Reduce `dropout` if underfitting
- Check data preprocessing with `test_small_networks.py`

**Poor Performance:**
- Verify `config.yaml` settings match your data
- Run `outlier_analysis.py` to find problematic graphs
- Enable `temperature_scaling` for calibrated predictions

---

## 📝 Notes

- All random seeds set to 42 for reproducibility (configurable)
- Results saved with timestamps to avoid overwrites
- GPU acceleration available if CUDA is detected
- Verbose logging available via `verbose: true` in config

---

## 📄 License



---

## 👤 Author

Simran Chauhan 

