import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
    )
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModeltrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class  ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModeltrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split train and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                train_array[:,:-1],
                train_array[:,-1]

            )
            
            models={
                "linear regression":LinearRegression(),
                "random forest":RandomForestRegressor(),
                "xgboost":XGBRegressor(),
                "cat boost":CatBoostRegressor(),
                "ada boost":AdaBoostRegressor(),
                "gradient boost":GradientBoostingRegressor(),
                "decision tree":DecisionTreeRegressor(),
                "kneighbours":KNeighborsRegressor(),
                
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("no best model found")
            
            logging.info("best found model on both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predict=best_model.predict(X_test)
            r2_scores = r2_score(y_test,predict)
            
            return r2_scores,best_model_name
        except Exception as e:
            raise CustomException(e,sys)