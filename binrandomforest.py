import logging
import pandas as pd

from math import ceil, sqrt
from random import randint
from typing import Literal, Tuple

from bintreepredictor import BinTreePredictor
from data import DataSet
from utils import round_wrp


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
            max_features :int|Literal['sqrt']|None=None,
            max_thresholds :int|None=None,
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
 
        self.trees :list[BinTreePredictor] = list()
        return


    def fit(self, data :DataSet) -> float:
        if self.max_features == "sqrt":
            self.max_features = ceil(sqrt(data.schema.num_features))

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
            for i in range(0, self.num_trees)
        ]

        for tree in self.trees:
            ds = data.sample(frac=1, replace=True, seed=randint(1, 2**30))
            tree.fit(ds)

        _, train_err = self.predict(data)
        logger.info(f"BinRandomForest_id:{self.id} - training_err:{round_wrp(train_err, 4)}")
        return train_err


    def predict(self, data :DataSet) -> Tuple[pd.Series, float|None]:
        predictions = pd.DataFrame(index=data.index)
        l = BinTreePredictor.LOSS_FUNC[self.loss_func_name]

        for i, tree in enumerate(self.trees):
            tree_pred, _ = tree.predict(data)
            predictions.insert(i, i, tree_pred)

        predictions = predictions.mode(axis="columns").iloc[:, 0]
        test_err = l(data.get_labels_as_series(), predictions) / len(data) if data.schema.has_labels() else None
        
        logger.info(f"BinRandomForest_id:{self.id} - test_err:{round_wrp(test_err, 4)}")
        return predictions, test_err


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