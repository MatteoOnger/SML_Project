import logging
import numpy as np
import pandas as pd

from dataclasses import dataclass
from math import log2
from typing import Any, Literal, Tuple

from data import DataSet, DataType
from errors import InvalidOperationError
from utils import round_wrp



logger = logging.getLogger(__name__)



class BinNode():
    """
    """
    def __init__(self, parent :'BinNode|None'=None, tree :'BinTreePredictor|None'=None, id :int=0) -> None:
        self.id = id

        self.isleaf = True
        self.tree = tree
        self.depth = parent.depth + 1 if parent is not None else 0

        self.parent = parent
        self.sx = None
        self.dx = None

        self.test = None

        self.data = None
        self.ispure = None
        self.prediction = None
        return


    def check_test(self, data :DataSet) -> np.ndarray:
        if self.test is None:
            raise InvalidOperationError("No test associated with this node")
        if self.test.feature not in data.schema.features:
            raise ValueError(f"Dataset doesn't have feature {self.test.feature}")

        op :callable
        res :pd.Series
        dt = data.schema.get_type(self.test.feature)

        if dt == DataType.CATEGORICAL:
            op = pd.Series.__eq__
        elif dt == DataType.NUMERICAL:
            op = pd.Series.__le__
        else:
            raise NotImplementedError(f"Method not implemented for {dt} features")
        
        try:
            res = op(data.get_feature_as_series(self.test.feature), self.test.threshold)
        except TypeError:
            raise TypeError("Mismatching types between feature and threshold") from None
        return res.to_numpy()


    def drop_data(self) -> None:
        self.data = None
        self.ispure = None
        return


    def drop_test(self) -> None:
        self.test = None
        return


    def predict(self, data :DataSet) -> pd.Series:
        if self.prediction is None and self.isleaf:
            raise InvalidOperationError("No prediction set fot this node")
        
        pred :pd.Series
        if self.isleaf:
            pred = pd.Series([self.prediction] * len(data), index=data.index)
        else:
            res = self.check_test(data)
            sx_pred = self.sx.predict(data[res])
            dx_pred = self.dx.predict(data[[not i for i in res]])
            pred = pd.concat([sx_pred, dx_pred]).sort_index()
        return pred


    def set_data(self, data :DataSet) -> None:
        if not self.isleaf:
            raise InvalidOperationError("Inner nodes can NOT contain datapoints")
        labels = data.get_labels_as_series()
        self.data = data
        self.ispure = bool((labels == labels.iloc[0]).all())
        self.prediction = self.tree.prediction_criterion(labels) if self.tree is not None else None
        return


    def set_test(self, feat :int|str, threshold :Any) -> None:
        self.test = BinNode.TestCondition(feat, threshold)
        return


    def split_node(self, feat :int|str, threshold :Any) -> None:
        if not self.isleaf:
            raise InvalidOperationError("Inner nodes can NOT be split")
        
        self.isleaf = False
        self.set_test(feat, threshold)
        res = self.check_test(self.data)

        self.sx = BinNode(self, self.tree, 2 * self.id + 1)
        self.dx = BinNode(self, self.tree, 2 * self.id + 2)

        self.sx.set_data(self.data[res])
        self.dx.set_data(self.data[[not i for i in res]])
        
        self.data = None
        self.ispure = None
        self.prediction = None
        return


    def __str__(self) -> str:
        s = {
            "id": self.id,
            "type": "leaf" if self.isleaf else "inner",
            "tree_id":  None if self.tree is None else self.tree.id,
            "depth": self.depth,
            "parent_id": None if self.parent is None else self.parent.id,
            "sx_id": None if self.sx is None else self.sx.id,
            "dx_id": None if self.dx is None else self.dx.id,
            "test": None if self.test is None else f"({self.test.feature}, {round_wrp(self.test.threshold, 4)})",
            "prediction": self.prediction,
            "has_data": len(self.data) if self.data is not None else None,
            "is_pure": self.ispure
        }
        return "node -> " + str(s)


    @dataclass
    class TestCondition():
        feature :int|str
        threshold :Any



class BinTreePredictor():
    """
    """
    LOSS_FUNC = {
        "zero-one": lambda y, p: (p != y).sum()
    }


    PREDICTION_CRITERION = {
        "mode": lambda y: y.mode()[0],
    }


    SPLIT_CRITERION = {
        "entropy": lambda y: y.value_counts(normalize=True).apply(lambda x: -x * log2(x)).sum() / max(log2(len(y.unique())), 1),
        "gini": lambda y: len(y.unique()) * (1 - y.value_counts(normalize=True).apply(lambda x: x**2).sum()) / max((len(y.unique()) - 1), 1),
        "misclass": lambda y: 1 - y.value_counts(normalize=True).max(),
    }


    STOP_CRITERION = {
        "max_nodes": lambda tree, n: tree.num_nodes > n,
        "max_height": lambda tree, h: tree.height > h,
    }


    def __init__(
            self,
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

        self.loss_func_name = loss_func
        self.prediction_criterion_name = prediction_criterion
        self.split_criterion_name = split_criterion
        self.stop_criterion_name = stop_criterion

        self.loss_func = BinTreePredictor.LOSS_FUNC[loss_func]
        self.prediction_criterion = BinTreePredictor.PREDICTION_CRITERION[prediction_criterion]
        self.split_criterion = BinTreePredictor.SPLIT_CRITERION[split_criterion]
        self.stop_criterion =  BinTreePredictor.STOP_CRITERION[stop_criterion]
        self.stop_threshold = stop_criterion_threshold
        
        self.max_features = max_features
        self.max_thresholds = max_thresholds

        self.num_nodes = 0
        self.height = 0

        self.root :BinNode = None
        self.leaves :list[BinNode] = list()
        return


    def fit(self, data :DataSet) -> float:
        root = BinNode(parent=None, tree=self)
        root.set_data(data)

        self.root = root
        self.leaves.append(root)

        self.num_nodes = 1
        self.height = 1

        while not self.stop_criterion(self, self.stop_threshold):
            best_leaf = None
            best_info_gain = 0
            best_feat, best_threshold = None, None

            for leaf in self.leaves:
                if leaf.ispure:
                    continue

                info_parent = self.split_criterion(leaf.data.get_labels_as_series())

                features = leaf.data.schema.features.to_numpy()
                if self.max_features is not None:
                    np.random.shuffle(features)
                    features = features[:self.max_features]
                
                for feat in features:                    
                    thresholds :np.ndarray
                    if self.max_thresholds is not None and leaf.data.schema.get_type(feat) == DataType.NUMERICAL:
                        _, thresholds = pd.cut(leaf.data.get_feature_as_series(feat), bins=self.max_thresholds + 1, retbins=True)
                        thresholds = thresholds[1:-1]
                    else:
                        thresholds = leaf.data.schema.get_feature_domain(feat)
                        if len(thresholds) == 1:
                            continue

                    for threshold in thresholds:
                        leaf.set_test(feat, threshold)
                        res = leaf.check_test(leaf.data)

                        data_sx = leaf.data[res]
                        data_dx = leaf.data[[not i for i in res]]
                        
                        leaf.drop_test()
                        if len(data_sx) == 0 or len(data_dx) == 0:
                            continue

                        info_sx = self.split_criterion(data_sx.get_labels_as_series())
                        info_dx = self.split_criterion(data_dx.get_labels_as_series())
                        info_gain = info_parent - (len(data_sx) * info_sx + len(data_dx) * info_dx) / len(leaf.data)

                        if info_gain > best_info_gain:
                            best_leaf = leaf
                            best_info_gain = info_gain
                            best_feat, best_threshold = feat, threshold
                        
            if best_info_gain > 0:
                logger.info(f"BinTreePredictor_id:{self.id} - split:(leaf:{best_leaf.id}, feat:{best_feat}, threshold:{round_wrp(best_threshold, 4)}) - info_gain:{round_wrp(best_info_gain, 4)}")

                best_leaf.split_node(best_feat, best_threshold)

                self.leaves.remove(best_leaf)
                self.leaves.append(best_leaf.sx)
                self.leaves.append(best_leaf.dx)

                self.num_nodes += 2
                self.height = max(self.height, best_leaf.sx.depth+1)
            else:
                logger.info(f"BinTreePredictor_id:{self.id} - no split found")
                break
        
        train_err = 0
        for leaf in self.leaves:
            train_err += self.loss_func(leaf.data.get_labels_as_series(), leaf.prediction)
        train_err /= len(data)

        logger.info(f"BinTreePredictor_id:{self.id} - training_err:{round_wrp(train_err, 4)}")
        return train_err


    def predict(self, data :DataSet) -> Tuple[DataSet, float|None]:
        if self.root is None:
            raise InvalidOperationError("This method cannot be called on an untrained predictor")
        predictions = self.root.predict(data)
        test_err = self.loss_func(data.get_labels_as_series(), predictions) / len(data) if data.schema.has_labels() else None
        return predictions, test_err


    def print_tree(self) -> None:
        print(f"\n---- TREE ----\n{self}\n")

        if self.root is not None:
            print(self.root)

        def rec(node :BinNode) -> None:
            if node.sx is not None:
                print(node.sx)
            if node.dx is not None:
                print(node.dx)
            
            if node.sx is not None:
                rec(node.sx)
            if node.dx is not None:
                rec(node.dx)
                return
        
        rec(self.root)
        print("---- ---- ----")
        return


    def __str__(self) -> str:
        s = {
            "id": self.id,
            "prediction_criterion": self.prediction_criterion_name,
            "split_criterion": self.split_criterion_name,
            "stop_criterion": f"({self.stop_criterion_name}, {self.stop_threshold})",
            "num_nodes": self.num_nodes,
            "height": self.height,
            "max_features": self.max_features,
            "max_thresholds": self.max_thresholds,
            "root": self.root.id,
            "leaves": [leaf.id for leaf in self.leaves]
        }
        return "tree -> " + str(s)