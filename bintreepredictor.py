import numpy as np
import pandas as pd

from dataclasses import dataclass
from math import log2
from typing import Any, Literal, Tuple

from data import DataSet, DataType
from errors import InvalidOperationError



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
        self.prediction = None
        return


    def check_test(self, data :DataSet) -> np.ndarray:
        if self.test is None:
            raise InvalidOperationError("No test associated with this node")
        if self.test.feature not in data.schema.features:
            raise ValueError("Dataset doesn't have feature {self.test.feature}")

        op :callable
        res :pd.Series
        dt = data.schema.get_type(self.test.feature)

        if dt == DataType.CATEGORICAL:
            op = pd.Series.__eq__
        elif dt == DataType.NUMERICAL:
            op = pd.Series.__le__
        else:
            raise NotImplementedError("Method not implemented for {dt} features")
        
        try:
            res = op(data.get_feature_as_series(self.test.feature), self.test.threshold)
        except TypeError:
            raise TypeError("Mismatching types between feature and threshold") from None
        return res.to_numpy()


    def drop_data(self) -> None:
        self.data = None
        return


    def drop_test(self) -> None:
        self.test = None
        return


    def is_pure(self) -> bool:
        if not self.isleaf:
            raise InvalidOperationError("Inner nodes can NOT contain datapoints")
        if self.data is None:
            raise InvalidOperationError("No data assigned to this leaf")
        labels = self.data.get_labels_as_series().to_numpy()
        return (labels[0] == labels).all()


    def predict(self, data :DataSet) -> pd.Series:
        if self.prediction is None and self.isleaf:
            raise InvalidOperationError("No prediction set fot this node")
        
        pred :pd.Series
        if self.isleaf:
            pred = pd.Series([self.prediction] * len(data), index=data.index)
        else:
            res = self.check_test(data)
            print(res)
            sx_pred = self.sx.predict(data[res])
            dx_pred = self.dx.predict(data[[not i for i in res]])
            pred = pd.concat([sx_pred, dx_pred]).sort_index()
        return pred


    def set_data(self, data :DataSet) -> None:
        if not self.isleaf:
            raise InvalidOperationError("Inner nodes can NOT contain datapoints")
        self.data = data
        self.prediction = self.tree.prediction_criterion(data.get_labels_as_series()) if self.tree is not None else None
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
            "test": None if self.test is None else f"({self.test.feature}, {self.test.threshold})",
            "prediction": self.prediction,
            "has_data": self.data is not None,
        }
        return "node -> " + str(s)


    @dataclass
    class TestCondition():
        feature :int|str
        threshold :Any



class BinTreePredictor():
    """
    """
    PREDICTION_CRITERION = {
        "mode": lambda y: y.mode()[0],
    }


    SPLIT_CRITERION = {
        "entropy": lambda y, t: y.value_counts(normalize=True).apply(lambda x: -x * log2(x)).sum() / max(log2(t), 1),
        "gini": lambda y, t: (1 - y.value_counts(normalize=True).apply(lambda x: x**2).sum()) / (max(t-1, 1) / t),
        "misclass": None,
    }


    #TODO: struct risk min
    STOP_CRITERION = {
        "max_nodes": lambda tree, n: tree.num_nodes > n,
        "max_height": lambda tree, h: tree.height > h,
    }


    def __init__(
            self,
            prediction_criterion :Literal['mode'],
            split_criterion :Literal['entropy', 'gini'],
            stop_criterion :Literal['max_nodes', 'max_height'],
            stop_criterion_threshold :int,
            id :int=0
        ) -> None:
        self.id = id

        self.prediction_criterion_name = prediction_criterion
        self.split_criterion_name = split_criterion
        self.stop_criterion_name = stop_criterion

        self.prediction_criterion = BinTreePredictor.PREDICTION_CRITERION[prediction_criterion]
        self.split_criterion = BinTreePredictor.SPLIT_CRITERION[split_criterion]
        self.stop_criterion =  BinTreePredictor.STOP_CRITERION[stop_criterion]
        self.stop_threshold = stop_criterion_threshold

        self.num_nodes = 0
        self.height = 0

        self.root :BinNode = None
        self.leaves :list[BinNode] = list()
        return


    def fit(self, data :DataSet) -> float:
        while not self.stop_criterion(self, self.stop_threshold):
            if self.root == None:
                root = BinNode(parent=None, tree=self)
                root.set_data(data)

                self.root = root
                self.leaves.append(root)

                self.num_nodes += 1
                self.height = 1
            else:
                best_leaf, best_feat, best_value, best_loss = None, None, None, float("inf")

                for leaf in self.leaves:
                
                    if leaf.is_pure():
                        continue
                
                    for feat in data.schema.features:
                        for value in data.schema.get_feature_domain(feat):
                            leaf.set_test(feat, value)
                            res = leaf.check_test(leaf.data)

                            data_sx = leaf.data[res]
                            data_dx = leaf.data[[not i for i in res]]
                            
                            if len(data_sx) == 0 or len(data_dx) == 0:
                                continue

                            loss_sx = self.split_criterion(data_sx.get_labels_as_series(), len(data_sx))
                            loss_dx = self.split_criterion(data_dx.get_labels_as_series(), len(data_dx))

                            loss = (loss_sx * len(data_sx) + loss_dx * len(data_dx)) / len(leaf.data)

                            if loss < best_loss:
                                best_leaf = leaf
                                best_feat = feat
                                best_value = value
                                best_loss = loss
                            
                            leaf.drop_test()

                best_leaf.split_node(best_feat, best_value)

                self.leaves.remove(best_leaf)
                self.leaves.append(best_leaf.sx)
                self.leaves.append(best_leaf.dx)

                self.num_nodes += 2
                self.height = max(self.height, best_leaf.sx.depth+1)
        return


    def predict(self, data :DataSet) -> Tuple[DataSet, float|None]:
        if self.root is None:
            raise InvalidOperationError("This method cannot be called on an untrained predictor")
        predictions = self.root.predict(data)
        accuracy = (predictions == data.get_labels_as_series()).sum() / len(data) if data.schema.has_labels() else None
        return predictions, accuracy


    def __str__(self) -> str:
        s = {
            "id": self.id,
            "prediction_criterion": self.prediction_criterion_name,
            "split_criterion": self.split_criterion_name,
            "stop_criterion": f"({self.stop_criterion_name}, {self.stop_threshold})",
            "num_nodes": self.num_nodes,
            "height": self.height,
            "root": self.root.id,
            "leaves": [leaf.id for leaf in self.leaves]
        }
        return "tree -> " + str(s)