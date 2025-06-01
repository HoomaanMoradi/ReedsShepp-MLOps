from config_reader import read_config
from data_ingestion import DataIngestion
from data_processing import DataProcessing
from model_training import ModelTraining

data_ingestion = DataIngestion(read_config("config/config.yaml"))
data_ingestion.run()

data_processing = DataProcessing(read_config("config/config.yaml"))
data_processing.run()

model_training = ModelTraining(read_config("config/config.yaml"))
model_training.run()
