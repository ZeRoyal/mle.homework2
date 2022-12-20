import os
import shutil

from pyspark.ml.feature import HashingTF, IDF, IDFModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pyspark.mllib.linalg.distributed import MatrixEntry, CoordinateMatrix
import numpy as np

import traceback
import configparser
from adapter import SparkAdapter
from logger import Logger

SHOW_LOG = True

class Processor():

    def __init__(self):
        """
        init with config
        """
        self.config = configparser.ConfigParser()
        self.log = Logger(SHOW_LOG).get_logger(__name__)
        self.config_path = os.path.join(os.getcwd(), 'config.ini')
        self.config.read(self.config_path)

        # Spark Adapter with context and session
        try:
            self.adapter = SparkAdapter()
            self.sc = self.adapter.get_context()
            self.spark = self.adapter.get_session()
        except:
            self.log.error(traceback.format_exc())

        # load models
        if not self.get_all_models():
            raise Exception('Can\'t load models')
        
        self.log.info("Processor is ready")
        pass

    def get_watched_matrix(self) -> bool:
        """
        load watched matrix
        """
        path = self.config.get("MODEL", "WATCHED_PATH")
        if path is None or not os.path.exists(path):
            self.log.error('Matrix of watched movies doesn\'t exists')
            return False
        
        self.log.info(f'Reading {path}')
        try:
            self.watched = CoordinateMatrix(self.spark.read.parquet(path) \
                .rdd.map(lambda row: MatrixEntry(*row)))
        except:
            self.log.error(traceback.format_exc())
            return False
        return True
    
    def get_tf(self) -> bool:
        """
        tf loading
        """
        path = self.config.get("MODEL", "TF_PATH")
        if path is None or not os.path.exists(path):
            self.log.error('TF model doesn\'t exists')
            return False
        
        self.log.info(f'Reading {path}')
        try:
            self.hashingTF = HashingTF.load(path)
        except:
            self.log.error(traceback.format_exc())
            return False
        return True
    
    def get_idf(self) -> bool:
        """
        get idf
        """

        path = self.config.get("MODEL", "IDF_PATH")
        if path is None or not os.path.exists(path):
            self.log.error('IDF model doesn\'t exists')
            return False
        
        self.log.info(f'Reading {path}')
        try:
            self.idf = IDFModel.load(path)
        except:
            self.log.error(traceback.format_exc())
            return False
        return True


    def recommend_check(self):
        """
        check for random user
        """

        self.log.info('Sample existing user recomendation')

        pass