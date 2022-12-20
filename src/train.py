import os
import shutil

from pyspark.ml.feature import HashingTF, IDF
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pyspark.mllib.linalg.distributed import MatrixEntry, CoordinateMatrix

from adapter import SparkAdapter
import traceback
import configparser
from logger import Logger

SHOW_LOG = True

class Trainer():

    def __init__(self) -> None:
        """
        init constructor with config
        """
        self.config = configparser.ConfigParser()
        self.log = Logger(SHOW_LOG).get_logger(__name__)
        self.config_path = os.path.join(os.getcwd(), 'config.ini')
        self.config.read(self.config_path)
        self.models_watched_path = './models/WATCHED'
        self.log.info("Trainer is ready")
        pass

    def preprocess(self, grouped) -> bool:
        """
        Create matrix for films have been watched
        """
        if not self.is_model_removed(self.models_watched_path):
            return False

        watched_matrix = CoordinateMatrix(grouped.flatMapValues(lambda x: x).map(lambda x: MatrixEntry(x[0], x[1], 1.0)))
        
        try:
            watched_matrix.entries.toDF().write.parquet(self.models_watched_path)
            self.config["MODEL"]["WATCHED_PATH"] = self.models_watched_path
            self.log.info(f"Watched movies are stored in {self.models_watched_path}")
        except:
            self.log.error(traceback.format_exc())
            return False

        return os.path.exists(self.models_watched_path)
    
    def make_tf(self, grouped, path='./models/TF_MODEL'):
        """
        Creating tf model
        """
        if not self.is_model_removed(path):
            return None

        df = grouped.toDF(schema=["user_id", "movie_ids"])

        FEATURES_COUNT = self.config.getint("MODEL", "FEATURES_COUNT", fallback=10000)
        self.log.info(f'TF-IDF features count = {FEATURES_COUNT}')

        # считаем TF - частоту токенов (фильмов), должна быть 1, т.к. пользователь либо посмотрел, либо не посмотрел фильм
        hashingTF = HashingTF(inputCol="movie_ids", outputCol="rawFeatures", numFeatures=FEATURES_COUNT)
        tf_features = hashingTF.transform(df)

        try:
            hashingTF.write().overwrite().save(path)
            self.config["MODEL"]["TF_PATH"] = path
            self.log.info(f"TF model stored at {path}")
        except:
            self.log.error(traceback.format_exc())
            return None
        
        return tf_features


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

    def write_idf(self, idf_features, path='./models/IDF_FEATURES') -> bool:
        '''
        idf write with config section
        '''
        if not self.is_model_removed(path):
            return False

        try:
            idf_features.write.format("parquet").save(path, mode='overwrite')
            self.config["MODEL"]["IDF_FEATURES_PATH"] = path
            self.log.info(f"IDF features stored at {path}")
        except:
            self.log.error(traceback.format_exc())
            return False
        return True

    
    def make_idf(self, tf_features, path='./models/IDF_MODEL') -> bool:
        """
        idf setuping
        """
        if not self.is_model_removed(path):
            return False
        idf = IDF(inputCol="rawFeatures", outputCol="features")
        idf = idf.fit(tf_features)

        self.log.info(f"IDF model type: {type(idf)}")

        try:
            idf.write().overwrite().save(path)
            self.config["MODEL"]["IDF_PATH"] = path
            self.log.info(f"IDF model stored at {path}")
        except:
            self.log.error(traceback.format_exc())
            return False

        # get features for users, saving them
        idf_features = idf.transform(tf_features)
        if not self.write_idf(idf_features):
            return False
        
        return True

    def train(self, input_filename=None) -> bool:
        '''
        main training method
        '''
        try:
            adapter = SparkAdapter()
            sc = adapter.get_context()
            spark = adapter.get_session()
        except:
            self.log.error(traceback.format_exc())
            return False
        
        if input_filename is None:
            INPUT_FILENAME = self.config.get("DATA", "INPUT_FILE", fallback="./data/sample.csv")
        else:
            INPUT_FILENAME = input_filename
        self.log.info(f'train data filename = {INPUT_FILENAME}')

        # reading file with group bying by users
        grouped = sc.textFile(INPUT_FILENAME, adapter.num_parts) \
            .map(lambda x: map(int, x.split())).groupByKey() \
            .map(lambda x : (x[0], list(x[1])))
        
        # watched films > matrix
        self.log.info('Calculating matrix of watched movies')
        if not self.preprocess(grouped):
            return False
        
        # getting features by tf
        self.log.info('Train TF model')
        tf_features = self.make_tf(grouped)
        if tf_features is None:
            return False
        
        # making idf
        self.log.info('Train IDF model')
        if not self.make_idf(tf_features):
            return False

        os.remove(self.config_path)
        with open(self.config_path, 'w') as configfile:
            self.config.write(configfile)       
        return True

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()