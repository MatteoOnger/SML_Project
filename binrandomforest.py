import logging
import pandas as pd

from math import ceil, sqrt
from random import randint
from typing import Literal, Tuple

from bintreepredictor import BinTreePredictor
from data import DataSet
from errors import InvalidOperationError
from utils import round_wrp



logger = logging.getLogger(__name__)



class BinRandomForest():
    """
    This class implements a random forest based on
    the binary decision trees implemented by the ``bintreepredictor.BinTreePredictor`` class.
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
        """
        Parameters
        ----------
        num_trees : int
            _description_
        loss_func : Literal['zero-one']
            Name of the loss function used to compute the training and the test error.
        prediction_criterion : Literal['mode']
            Name of the criterion used to assign a label to each leaf. 
        split_criterion : Literal['entropy', 'gini', 'misclass']
            Name of the criterion used to determine the best split.
        stop_criterion : Literal['max_nodes', 'max_height']
            Name of the criterion used to limit the growth of the decision trees.
        stop_criterion_threshold : int
            Threshold of the criterion used to limit the growth of the decision trees.
        max_features : int | Literal['sqrt'] | None, optional
            Max number of features considered per leaf during the search for the best split, by default None.
            If it is set to 'sqrt', the square root of the number of features in the training set 
            will be calculated automatically and used in training.
        max_thresholds : int | None, optional
            Max number of thresholds considered per feature and leaf during the search for the best split, by default None.
            This parameter is applied exclusively to numerical features.
        id : int, optional
            A unique id to identify the forest, by default 0.
        """
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
        """
        Trains the decision trees of the forest using the data provided.

        Parameters
        ----------
        data : DataSet
            Data points used to train the predictor.

        Returns
        -------
        :float
            Training error.

        Raises
        ------
        ValueError
            If the labels are not known for these data points.
        """
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
            ds = data.sample(frac=1, replace=True, ignore_index=True, seed=randint(1, 2**30))
            tree.fit(ds)

        _, train_err = self.predict(data)
        logger.info(f"BinRandomForest_id:{self.id} - training_err:{round_wrp(train_err, 4)}")
        return train_err


    def predict(self, data :DataSet) -> Tuple[pd.Series, float|None]:
        """
        Predicts the labels of the given data points.

        Parameters
        ----------
        data : DataSet
            Data points to process.

        Returns
        -------
        :Tuple[pd.Series, float|None]
            A series containing the predicted label for each data point provided.
            If ``data`` contains the expected labels, then the second entry of the tuple is the test error,
            otherwise it is None.

        Raises
        ------
        InvalidOperationError
            If the random forest is not trained.
        """
        if len(self.trees) == 0:
            raise InvalidOperationError("This method cannot be called on an untrained predictor")
        
        predictions = pd.DataFrame(index=data.index)
        loss = BinTreePredictor.LOSS_FUNC[self.loss_func_name]

        for i, tree in enumerate(self.trees):
            tree_pred, _ = tree.predict(data)
            predictions.insert(i, i, tree_pred)

        predictions = predictions.mode(axis="columns").iloc[:, 0]
        test_err = loss(data.get_labels_as_series().sort_index(), predictions) / len(data) if data.schema.has_labels() else None
        
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