![Python Version](https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white)
![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)
![code style: black](https://img.shields.io/badge/code%20style-black-black)
![MLflow](https://img.shields.io/badge/MLflow-tracking-blue?logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/api-FastAPI-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/docker-ready-blue?logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/kubernetes-deployed-blue?logo=kubernetes&logoColor=white)

# 🚀 ReedsShepp-MLOps

An end-to-end MLOps implementation for the Reeds-Shepp path planning algorithm, featuring automated data ingestion, processing, model training, and deployment pipelines. This project demonstrates industry best practices for building maintainable and scalable ML systems.

## 📌 Features

### Core Functionality

- **Automated ML Pipeline**: End-to-end workflow from data ingestion to model deployment
- **Cloud-Native**: Seamless integration with Google Cloud Storage for data management
- **Experiment Tracking**: MLflow integration for tracking experiments, parameters, and metrics
- **Reproducible Training**: Versioned data, code, and configurations for full reproducibility
- **Modular Design**: Clean separation of concerns with well-defined interfaces

### Technical Highlights

- **Robust Configuration**: YAML-based configuration system with validation
- **Comprehensive Logging**: Structured logging for better observability
- **Type Safety**: Full type hints throughout the codebase
- **Testing**: Unit tests for critical components
- **Documentation**: Comprehensive docstrings and developer guides


## 🚀 Getting Started

### Prerequisites

- Python 3.10.12
- Google Cloud account (for GCS integration)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ReedsShepp-MLOps.git
   cd ReedsShepp-MLOps
   ```

2. **Set up the virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r dev-requirements.txt  # For development
   ```

3. **Configure Google Cloud** (if using GCS)
   ```bash
   # Install Google Cloud SDK
   gcloud auth login
   gcloud config set project your-project-id
   
   # Set up application default credentials
   gcloud auth application-default login
   ```

## 🚦 Usage

### Running the Pipeline

1. **Update the configuration**
   Edit `config/config.yaml` with your project-specific settings:
   ```yaml
   data_ingestion:
     project_id: your-project-id
     bucket_name: your-bucket-name
     train_val_object_name: data/train_val.npz
     test_object_name: data/test.npz
     
   model_training:
     max_iter: 1000
     hidden_layer_sizes: [50, 50]
     learning_rate_init: 0.001
   ```

2. **Run the complete ML pipeline**
   ```bash
   python src/main.py
   ```

3. **Monitor the pipeline**
   - Check the console output for real-time logs
   - View detailed logs in `logs/` directory
   - Access MLflow UI: `mlflow ui` (if enabled)

## 📂 Project Structure

```
ReedsShepp-MLOps/
├── .github/                 # GitHub workflows and templates
├── config/
│   └── config.yaml         # Main configuration file
│
├── logs/                    # Application logs
│
├── models/                  # Trained model artifacts
│
├── src/                     # Source code
│   ├── __init__.py          # Package initialization
│   ├── config_reader.py     # Configuration management
│   ├── data_ingestion.py    # Data loading and validation
│   ├── data_processing.py   # Data preprocessing and feature engineering
│   ├── logger.py            # Logging configuration
│   ├── model_training.py    # Model training and evaluation
│   └── main.py              # Pipeline orchestration
│
├── .env.example           # Example environment variables
├── .gitignore
├── .pre-commit-config.yaml  # Pre-commit hooks
├── pyproject.toml          # Project metadata and dependencies
├── requirements.txt        # Production dependencies
├── dev-requirements.txt    # Development dependencies
└── README.md              # This file
```

## 🔧 Configuration

The application is configured through `config/config.yaml`. Key sections include:

### Data Ingestion
```yaml
data_ingestion:
  project_id: your-project-id          # GCP project ID
  bucket_name: your-bucket-name        # GCS bucket name
  train_val_object_name: data/train_val.npz  # Training/validation data path in GCS
  test_object_name: data/test.npz            # Test data path in GCS
  train_ratio: 0.8                     # Train/validation split ratio
  artifact_dir: artifacts/raw          # Local directory for downloaded data
```

### Model Training
```yaml
model_training:
  max_iter: 1000                     # Maximum training iterations
  random_state: 42                   # Random seed for reproducibility
  hidden_layer_sizes: [50, 50]     # Network architecture
  top_k: 5                          # Top-k metrics to track
  early_stop_number: 5              # Early stopping patience
  learning_rate_init: 0.001         # Initial learning rate
  model_name: "nn_50_50"          # Model name for tracking
```

## 🙏 Acknowledgment

This project builds upon and adapts code from the following open-source projects:
- [TAP30 Ride Demand MLOps](https://github.com/aaghamohammadi/tap30-ride-demand-mlops)

