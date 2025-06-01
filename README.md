# ReedsShepp-MLOps

An end-to-end MLOps implementation for the Reeds-Shepp path planning algorithm, featuring automated data ingestion, processing, model training, and deployment pipelines.

## 🚀 Features

- **ML Pipeline Automation**: Automated workflow from data ingestion to model training
- **Cloud Storage Integration**: Seamless data handling with Google Cloud Storage
- **Experiment Tracking**: MLflow integration for experiment tracking and model versioning
- **Configuration Management**: YAML-based configuration for easy parameter tuning
- **Modular Architecture**: Clean separation of concerns with dedicated modules

## 🛠️ Tech Stack

- **ML Framework**: Scikit-learn
- **Experiment Tracking**: MLflow
- **Cloud Storage**: Google Cloud Storage
- **API**: FastAPI
- **Configuration**: YAML
- **Dependency Management**: Poetry (pyproject.toml)

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ReedsShepp-MLOps.git
   cd ReedsShepp-MLOps
   ```

2. Set up a virtual environment and install dependencies:
   ```bash
   source .venv/bin/activate
   pip install -r dev-requirements.txt
   pip install -r requirements.txt
   ```

3. Configure Google Cloud credentials (if using GCS):

## 🚦 Usage

1. Update the configuration in `config/config.yaml` with your project-specific settings.

2. Run the complete ML pipeline:
   ```bash
   python src/main.py
   ```

## 📂 Project Structure

```
ReedsShepp-MLOps/
├── config/
│   └── config.yaml         # Configuration file
├── src/
│   ├── __init__.py
│   ├── config_reader.py   # Configuration management
│   ├── data_ingestion.py  # Data loading from GCS
│   ├── data_processing.py # Data preprocessing
│   ├── logger.py          # Logging utilities
│   ├── model_training.py  # Model training logic
│   └── main.py           # Pipeline orchestration
├── tests/                 # Unit tests
├── .gitignore
├── pyproject.toml         # Project metadata and dependencies
├── requirements.txt       # Production dependencies
└── dev-requirements.txt   # Development dependencies
```

## 🔧 Configuration

Modify `config/config.yaml` to adjust:
- Data source paths and parameters
- Model hyperparameters
- Training settings
- Cloud storage configurations

## 🙏 Acknowledgments

This project builds upon and adapts code from the following open-source project:
- [TAP30 Ride Demand MLOps](https://github.com/aaghamohammadi/tap30-ride-demand-mlops) - Used as a reference for MLOps pipeline implementation

