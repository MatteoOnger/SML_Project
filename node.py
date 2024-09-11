import numpy as np
import pandas as pd

import treepredictor as tp

from dataclasses import dataclass
from typing import Any

from data import DataType, DataSet
from errors import InvalidOperationError



class BinNode():
    """
    """
    def __init__(self, parent :'BinNode|None'=None, tree :tp.BinTreePredictor|None=None, id :int=0) -> None:
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
            op = pd.Series.__ne__
        elif dt == DataType.NUMERICAL:
            op = pd.Series.__lt__
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
        self.prediction = self.tree.prediction_criterion(data.get_labels_as_series())
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
            "parent_id": None if self.parent is None else self.parent.id,
            "sx_id": None if self.sx is None else self.sx.id,
            "dx_id": None if self.dx is None else self.dx.id,
            "test": None if self.test is None else f"({self.test.feature}, {self.test.threshold})",
            "prediction": self.prediction,
            "has_data": self.data is not None
        }
        return "node -> " + str(s)


    @dataclass
    class TestCondition():
        feature :int|str
        threshold :Any