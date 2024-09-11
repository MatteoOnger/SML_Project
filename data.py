import numpy as np
import pandas as pd

from enum import Enum



class DataType(Enum):
    """
    Enumeration of all the possible data types considered.
    """
    CATEGORICAL = 0
    NUMERICAL   = 1



class DataSet():
    """
    """
    def __init__(self, data :pd.DataFrame, label_col :str|None=None) -> None:
        if label_col is not None and label_col not in data.columns:
            raise ValueError(f"'{label_col}' is not a column of <data>")

        self.data = data
        self.index = data.index
        self.label_col = label_col

        self.schema = DataSet.Schema(self)
        return


    def get_feature_as_series(self, col :int|str) -> pd.Series:
        if col not in self.schema.features:
            raise ValueError(f"No feature named {col}")
        return self.data[col]


    def get_labels_as_series(self) -> pd.Series:
        if self.label_col is None:
            raise ValueError("No labels in this dataset")
        return self.data[self.label_col]


    def __getitem__(self, key :int|slice) -> 'DataSet':
        data = self.data[key]
        label_col = self.label_col if self.label_col in data.columns else None
        return DataSet(data, label_col)


    def __len__(self) -> int:
        return len(self.data)


    def __str__(self) -> str:
        return self.data.__str__()


    class Schema():
        """
        """
        def __init__(self, ds :'DataSet') -> None:
            self.ds = ds

            self.features = ds.data.columns.drop(self.ds.label_col) if self.ds.label_col is not None else ds.data.columns
            self.num_features = len(self.features)
            return


        def get_label_domain(self) -> np.ndarray:
            return self.ds.get_labels_as_series().unique()


        def get_feature_domain(self, col :int|str) -> np.ndarray:
            return self.ds.get_feature_as_series(col).unique()


        def get_type(self, col :int|str) -> DataType:
            return DataType.CATEGORICAL if self.ds.data.loc[:, col].dtype ==  object else DataType.NUMERICAL
        

        def has_labels(self) -> bool:
            return self.ds.label_col is not None