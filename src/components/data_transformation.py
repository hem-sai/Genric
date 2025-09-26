import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from  sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for the data transformation
        '''
        try:
            numerical_columns =  ['reading_score', 'writing_score']
            categorical_columns =  [
                'gender',
                'race_ethnicity', 
                'parental_level_of_education', 
                'lunch', 
                'test_preparation_course'
                ]
            
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]

            )
            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_object()
            target_column_name="math_score"
            numerical_columns = ['reading_score', 'writing_score'] #why?

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            logging.info("applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            '''
            fit = Learn something from the data (e.g., mean & std for scaling, min & max, encoding categories, etc.).
            transform = Apply that learned information to actually change the data.
            fit_transform = Does both in one go (first learns, then applies).
            ⚡ Rule of thumb:
            On training data → use fit_transform (learn + apply).
            On test data → use only transform (apply what you learned from train).
            👉 This prevents data leakage (you don’t want to “peek” at test data when learning scaling parameters,
              encodings, etc.).
            '''

            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            '''
            np.c_ is shorthand for column-wise concatenation.
            It sticks arrays side by side (like pd.concat(..., axis=1) but for NumPy).
            So here:
            input_feature_train_arr → preprocessed features (X).
            np.array(target_feature_train_df) → target values (y).
            np.c_ merges them into one big NumPy array → train_arr.
            '''

            logging.info("Saved Preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)