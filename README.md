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

- Python 3.10
- Google Cloud account (for GCS integration)

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/HoomaanMoradi/ReedsShepp-MLOps.git
   cd ReedsShepp-MLOps
   ```

2. **Set up the development environment**
   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install package in development mode
   pip install -e .
   ```

3. **Configure Google Cloud**
   ```bash
   # Install Google Cloud SDK
   gcloud auth login
   gcloud config set project your-project-id
   
   # Set up application default credentials
   gcloud auth application-default login
   ```

## 🛡️ Docker Requirements

When using Docker, it's essential to mount your own artifacts directory and GCP credentials:

```yaml
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ./artifacts:/app/artifacts  # Mount your own artifacts directory
      - ./gcp-credentials.json:/app/gcp-credentials.json  # Mount your own GCP credentials
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

2. **Run the complete ML pipeline**
   ```bash
   python pipeline/run.py
   ```

3. **Monitor the pipeline**
   - Check the console output for real-time logs
   - View detailed logs in `logs/` directory
   - Access MLflow UI: `mlflow ui`

4. **API**

The project includes a FastAPI-based REST API for demand prediction. The API is served at `http://0.0.0.0:8080` by default.

#### API Endpoints

- **GET `/docs`**: Interactive API

You can access the interactive API at: http://0.0.0.0:8080/docs

The documentation interface provides a form for making predictions with the following input fields:
- `input1`: Angle in radians (0 to π)
- `input2`: Angle in radians (-π to π)
- `input3`: Distance (0 to 8.5)

The API returns predictions with class indices and probabilities through the Swagger UI interface.

#### Prediction API Usage

The `/predict` endpoint accepts POST requests with the following JSON payload:

```json
{
    "input1": 1.0,    // Angle in radians (0 to π)
    "input2": 0.0,    // Angle in radians (-π to π)
    "input3": 8.5     // Distance (0 to 8.5)
}
```

The API returns a JSON response with the top-k predictions:

```json
{
    "predictions": [
        {
            "class_index": 0,
            "probability": 0.95
        },
        {
            "class_index": 1,
            "probability": 0.03
        },
        // ... additional predictions
    ]
}
```

#### Running the API Server

Start the FastAPI server using:
```bash
python web/application.py
```

The server will start on port 8080 by default. You can access the interactive API at:
- http://localhost:8080/docs


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
├── pipeline/                # Pipeline scripts and utilities
│   └── run.py              # Main pipeline execution script
│
├── src/                     # Source code
│   ├── __init__.py          # Package initialization
│   ├── config_reader.py     # Configuration management
│   ├── data_ingestion.py    # Data loading and validation
│   ├── data_processing.py   # Data preprocessing and feature engineering
│   ├── logger.py            # Logging configuration
│   ├── model_training.py    # Model training and evaluation
│
├── web/                     # Web application
│   └── application.py       # FastAPI application for predictions
│
├── .env.example           # Example environment variables
├── .gitignore
├── .pre-commit-config.yaml  # Pre-commit hooks
├── pyproject.toml          # Project metadata and dependencies
├── requirements.txt        # Production dependencies
├── dev-requirements.txt    # Development dependencies
├── Dockerfile              # Container configuration
├── docker-compose.yaml     # Multi-container orchestration
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

