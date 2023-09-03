import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','raw_data.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initialize_data_ingestion(self):
        logging.info("Enter the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/datasets/combined.csv')
            df['ds'] = pd.to_datetime(df['ds'])
            df = df[df['unique_id'] == 'H89']
            logging.info("Read dataset as dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, header=True, index=False)
            logging.info("Saved raw data to {}".format(self.ingestion_config.raw_data_path))
            
            df_train, df_test = df[df.ds <= '2019-11-01'], df[df.ds > '2019-11-01']
            df_train.to_csv(self.ingestion_config.train_data_path, header=True, index=False)
            df_test.to_csv(self.ingestion_config.test_data_path, header=True, index=False)
            logging.info("Saved train data to {}".format(self.ingestion_config.train_data_path))
            logging.info("Saved test data to {}".format(self.ingestion_config.test_data_path))
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data,test_data = data_ingestion.initialize_data_ingestion()