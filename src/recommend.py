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
    
    def get_idf_features(self) -> bool:
        """
        get idf features for users
        """        
        path = self.config.get("MODEL", "IDF_FEATURES_PATH")
        if path is None or not os.path.exists(path):
            self.log.error('IDF features doesn\'t exists')
            return False
        
        self.log.info(f'Reading {path}')
        try:
            self.idf_features = self.spark.read.load(path)
        except:
            self.log.error(traceback.format_exc())
            return False
        return True

    def get_all_models(self) -> bool:
        """
        Main loader
        """

        self.log.info('Loading watched movies')
        if not self.get_watched_matrix():
            return False
        
        self.log.info('Loading TF model')
        if not self.get_tf():
            return False

        self.log.info('Loading IDF model')
        if not self.get_idf():
            return False

        self.log.info('Loading IDF features')
        if not self.get_idf_features():
            return False

        return True
    
    def receive_recomend(self, ordered_similarity, max_count=5) -> list:
        """
        Get recommendations by similarity
        """        
        self.log.info('Calculate movies recommendations')
        users_sim_matrix = IndexedRowMatrix(ordered_similarity)
        multpl = users_sim_matrix.toBlockMatrix().transpose().multiply(self.watched.toBlockMatrix())        
        ranked_movies = multpl.transpose().toIndexedRowMatrix().rows.sortBy(lambda row: row.vector.values[0], ascending=False)

        result = []
        for i, row in enumerate(ranked_movies.collect()):
            if i >= max_count:
                break
            result.append((row.index, row.vector[0]))
        return result

    def recommend_check(self):
        """
        check for random user
        """

        self.log.info('Sample existing user recomendation')

        # get features - users matrix
        temp_matrix = IndexedRowMatrix(self.idf_features.rdd.map(
            lambda row: IndexedRow(row["user_id"], Vectors.dense(row["features"]))
        ))
        temp_block = temp_matrix.toBlockMatrix()

        self.log.info('Calculate similarities')
        similarities = temp_block.transpose().toIndexedRowMatrix().columnSimilarities()

        user_id = np.random.randint(low=0, high=self.watched.numCols())
        self.log.info(f'user ID: {user_id}')
        filtered = similarities.entries.filter(lambda x: x.i == user_id or x.j == user_id)

        # get users with the highest similarity
        ordered_similarity = filtered.sortBy(lambda x: x.value, ascending=False) \
            .map(lambda x: IndexedRow(x.j if x.i == user_id else x.i, Vectors.dense(x.value)))

        recomendations = self.receive_recomend(ordered_similarity)
        self.log.info('TOP recomendations for existing user:')
        for movie_id, rank in recomendations:
            self.log.info(f'- movie # {movie_id} (rank: {rank})')
        pass


if __name__ == "__main__":
    processor = Processor()
    processor.recommend_check()