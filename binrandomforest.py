import logging
import pandas as pd

from typing import Literal, Tuple

from bintreepredictor import BinTreePredictor
from data import DataSet


logger = logging.getLogger(__name__)



class BinRandomForest():
    """
    """
    def __init__(
            self,
            num_trees :int,
            loss_func :Literal['zero-one'],
            prediction_criterion :Literal['mode'],
            split_criterion :Literal['entropy', 'gini', 'misclass'],
            stop_criterion :Literal['max_nodes', 'max_height'],
            stop_criterion_threshold :int,
            max_features :int=None,
            max_thresholds :int=None,
            id :int=0
        ) -> None:
        self.id = id
        self.num_trees = num_trees

        self.loss_func_name = loss_func
        self.prediction_criterion_name = prediction_criterion
        self.split_criterion_name = split_criterion
        self.stop_criterion_name = stop_criterion
        
        self.stop_threshold = stop_criterion_threshold
        
        self.max_features = max_features
        self.max_thresholds = max_thresholds
 
        self.trees = [
            BinTreePredictor(
                self.loss_func_name,
                self.prediction_criterion_name,
                self.split_criterion_name,
                self.stop_criterion_name,
                self.stop_threshold,
                self.max_features,
                self.max_thresholds,
                i
            )
            for i in range(0, num_trees)
        ]
        return


    def fit(self, data :DataSet) -> float:
        pass


    def predict(self, data :DataSet) -> Tuple[pd.Series, float|None]:
        pass


    def __str__(self) -> str:
        s = {
            "id": self.id,
            "num_trees": self.num_trees,
            "prediction_criterion": self.prediction_criterion_name,
            "split_criterion": self.split_criterion_name,
            "stop_criterion": f"({self.stop_criterion_name}, {self.stop_threshold})",
            "max_features": self.max_features,
            "max_thresholds": self.max_thresholds,
        }
        return "random_forest -> " + str(s)