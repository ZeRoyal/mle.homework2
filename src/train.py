import os
import shutil
import sys
import argparse
from pyspark.ml.feature import HashingTF, IDF, IDFModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pyspark.mllib.linalg.distributed import MatrixEntry, CoordinateMatrix
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession
import numpy as np

import traceback
import configparser

from logger import Logger

SHOW_LOG = True

class TfIdf():

    def __init__(self) -> None:
        """
        init constructor with config
        """
        self.config = configparser.ConfigParser()
        self.log = Logger(SHOW_LOG).get_logger(__name__)
        self.config_path = os.path.join(os.getcwd(), 'config.ini')
        self.config.read(self.config_path)
        self.models_watched_path = './models/WATCHED_MATRIX_PATH'
        self.tfidf_path ='./models/TFIDF_FEATURES_PATH'
        self.log.info("TfIdf is ready")
        
        # Spark config from config.ini
        self.log.info("Config for Spark")
        self.config_s = SparkConf()
        self.config_s.set("spark.app.name", "mle.homework2")
        self.config_s.set("spark.master", "local")
        self.config_s.set("spark.executor.cores", \
            self.config.get("SPARK", "NUM_PROCESSORS", fallback="3"))
        self.config_s.set("spark.executor.instances", \
            self.config.get("SPARK", "NUM_EXECUTORS", fallback="1"))
        self.config_s.set("spark.executor.memory", "16g")
        self.config_s.set("spark.locality.wait", "0")
        self.config_s.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        self.config_s.set("spark.kryoserializer.buffer.max", "2000")
        self.config_s.set("spark.executor.heartbeatInterval", "6000s")
        self.config_s.set("spark.network.timeout", "10000000s")
        self.config_s.set("spark.shuffle.spill", "true")
        self.config_s.set("spark.driver.memory", "16g")
        self.config_s.set("spark.driver.maxResultSize", "16g")

        self.num_parts = self.config.getint("SPARK", "NUM_PARTS", fallback=None)
        self.log.info("Getting Spark config:")
        for conf in self.config_s.getAll():
            self.log.info(f'{conf[0].upper()} = {conf[1]}')
        
        # context and session
        self.sc = None
        self.spark = None

    def preprocess(self, grouped) -> bool:
        """
        Create matrix for films have been watched
        """
        if not self.is_model_removed(self.models_watched_path):
            return False

        self.watched_matrix = CoordinateMatrix(grouped.flatMapValues(lambda x: x) \
                                                        .map(lambda x: MatrixEntry(x[0], x[1], 1.0)))
        
        try:
            self.watched_matrix.entries.toDF().write.parquet(self.models_watched_path)
            self.config["MODEL"]["WATCHED_MATRIX_PATH"] = self.models_watched_path
            self.log.info(f"Watched movies are stored in {self.models_watched_path}")

        except:
            self.log.error(traceback.format_exc())
            return False

        return os.path.exists(self.models_watched_path)

    def is_model_removed(self, path) -> bool:
        """
        Try to remove old model and check if it was succeed
        """
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        if os.path.exists(path):
            self.log.error(f'Can\'t remove {path}')
            return False
        return True

    
    def tfIDF(self, df):
        '''
        create TF-IDF and save it for testing
        '''
        df = df.toDF(schema=['user_id', 'movie_ids'])
        hashingTF = HashingTF(inputCol="movie_ids", outputCol="rawFeatures", numFeatures=10000)
        tf = hashingTF.transform(df)
        tf.cache()

        idf = IDF(inputCol="rawFeatures", outputCol="features").fit(tf)

        # get features for users, saving them
        self.tf_idf = idf.transform(tf)
        if not self.write_tfidf():
            return False

        return True

    def write_tfidf(self) -> bool:
        '''
        idf write with config section
        '''
#        if not self.is_model_removed(self.tfidf_path):
#            return False

        try:
            self.tf_idf.write.format("parquet").save(self.tfidf_path, mode='overwrite')
            self.log.info(f"TFIDF saved to {self.tfidf_path}")
            self.config["MODEL"]["TFIDF_FEATURES_PATH"] = self.tfidf_path
        except:
            self.log.error(traceback.format_exc())
            return True
        return True

    def train(self, input_filename=None) -> bool:
        '''
        main training method
        '''        
        #Get spark context

        if self.sc is None:
            try:
                self.sc = SparkContext(conf=self.config_s)
                self.log.info("Initilizing SparkContext()")
            except:
                self.log.error(traceback.format_exc())
                return False


        #Get spark session

        if self.spark is None:
            try:
                self.spark = SparkSession(self.sc)
                self.log.info("Initilizing SparkSession()")
            except:
                self.log.error(traceback.format_exc())
                return False


        FILENAME = self.config.get("DATA", "INPUT_FILE", fallback="./data/sample.csv") if input_filename is None \
                                                                                                else input_filename
        # reading file with group bying by users
        spark_grouped_df = self.sc.textFile(FILENAME, self.config.getint("SPARK", "NUM_PARTS", fallback=None)) \
            .map(lambda x: map(int, x.split())).groupByKey() \
            .map(lambda x : (x[0], list(x[1])))
        
        # watched films > matrix
        self.log.info('Calculating matrix of watched movies')
        if not self.preprocess(spark_grouped_df):
            return False
        
        # making tfidf
        self.log.info('Get TF-IDF features')
        if not self.tfIDF(spark_grouped_df):
            return False

#        os.remove(self.config_path)
#        with open(self.config_path, 'w') as f:
#            print("PROBLEM===================")
#            self.config.write(f)

        return True

    def useTfIDF(self, is_training=True) -> bool:
        """
        get tfidf features for users
        """  
        #Get spark context

        if self.sc is None:
            try:
                self.sc = SparkContext(conf=self.config_s)
                self.log.info("Initilizing SparkContext()")
            except:
                self.log.error(traceback.format_exc())
                return False


        #Get spark session

        if self.spark is None:
            try:
                self.spark = SparkSession(self.sc)
                self.log.info("Initilizing SparkSession()")
            except:
                self.log.error(traceback.format_exc())
                return False
        path = self.config.get("MODEL", "TFIDF_FEATURES_PATH") 
        if not is_training:    
            if path is None or not os.path.exists(path):
                self.log.error(f'No TFIDF model! ({path})')
                return False
            else:
                try:
                    self.tf_idf = self.spark.read.load(path)
                except:
                    self.log.error(traceback.format_exc())
                    return False

        return True

    def use_w_matrix(self, is_training=True) -> bool:
        """
        load watched matrix
        """
        path = self.config.get("MODEL", "WATCHED_MATRIX_PATH")
        self.log.info(f'Reading watched_matrix from {path}')
        if not is_training: 
            if path is None or not os.path.exists(path):
                self.log.error(f'No watched matrix found ({path})')
            else:
                try:
                    self.watched_matrix = CoordinateMatrix(self.spark.read.parquet(path) \
                                                        .rdd.map(lambda row: MatrixEntry(*row)))
                except:
                    self.log.error(traceback.format_exc())
                    return False

        return True

    def get_all_models(self, is_training=True) -> bool:
        """
        Main loader
        """
        if not is_training:
            self.log.info('Loading TFIDF')
            if not self.useTfIDF(is_training=is_training):
                return False

            self.log.info('Loading matrix for watched movies')
            if not self.use_w_matrix(is_training=is_training):
                return False

        return True

    def recommend_check(self):
        """
        check for random user
        """

        self.log.info('Predict existing user recommendations')
        print(f"{self.tf_idf} WTF is =============")
        # get features - users matrix
        self.user_matrix = IndexedRowMatrix(self.tf_idf.rdd.map(
            lambda row: IndexedRow(row["user_id"], Vectors.dense(row["features"]))
        ))
        self.user_matrix_to_block = self.user_matrix.toBlockMatrix()
        self.log.info('Calculate user similarities')
        self.user_sim = self.user_matrix_to_block.transpose().toIndexedRowMatrix().columnSimilarities()

        random_user = np.random.randint(low=0, high=self.watched_matrix.numCols())
        filtered_user = self.user_sim.entries.filter(lambda y: y.i == random_user or y.j == random_user)
        print(f"====== !! {filtered_user} filtered_user ==============") 
        #print(f"====== !! {filtered_user.sortBy(lambda x: x.value, ascending=False).map(lambda x: IndexedRow(x.j if x.i == self.random_user else x.i, Vectors.dense(x.value)))} filtered_user2 ==============") 
        # get users with the highest similarity

        self.user_sim_ascending = filtered_user.sortBy(lambda x: x.value, ascending=False) \
            .map(lambda x: IndexedRow(x.j if x.i == random_user else x.i, Vectors.dense(x.value)))

        self.receive_recommend()
        return True



    def receive_recommend(self):
        """
        Get recommendations by similarity
        """  
        self.max_ = 10 
        self.log.info(f'Recommendations (top {self.max_}) for random user')     
        print(f"===== {self.tf_idf} ======================")
        print(f"====== !! {self.user_sim_ascending} ==============") 
        weights = IndexedRowMatrix(self.user_sim_ascending).toBlockMatrix().transpose() \
                                                        .multiply(self.watched_matrix.toBlockMatrix())        
        recommended_movies = weights.transpose().toIndexedRowMatrix().rows \
                                            .sortBy(lambda row: row.vector.values[0], ascending=False)

        for i, row in enumerate(recommended_movies.collect()):
            if i >= self.max_:
                break
            self.log.info(f'movie # {row.index} (rank: {row.vector[0]})')

        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predictor")
    parser.add_argument("-t",
                                 "--is_training",
                                 type=str,
                                 help="Select mode",
                                 required=True,
                                 default="train",
                                 const="train",
                                 nargs="?",
                                 choices=["train", "test"])
    args = parser.parse_args()
    model = TfIdf()
    if args.is_training == "train":
        model.train()
    else:
        model.get_all_models(is_training=False)
        model.recommend_check()
