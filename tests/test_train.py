
import sys
import os
sys.path.insert(1, os.path.join(os.getcwd(), "src"))
from adapter import SparkAdapter
from train import TfIdf

trainer = Trainer()

def test_train():
    assert trainer.train('./data/sample.csv')
