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
    This class implements a data set: each column represents a feature, or a label, while each row a data point.
    It wraps a pandas dataframe and provides some specific methods to simplify its use as training or test set.
    """
    def __init__(self, data :pd.DataFrame, label_col :str|None=None) -> None:
        """
        Parameters
        ----------
        data : pd.DataFrame
            The data points to process: know features plus, possibly, the expected label.
        label_col : str | None, optional
            The name of the column of ``data`` that contains the labels, by default None.

        Raises
        ------
        ValueError
            If ``label_col`` is not a column of ``data``.
        """
        if label_col is not None and label_col not in data.columns:
            raise ValueError(f"'{label_col}' is not a column of <data>")

        self.data = data
        self.index = data.index
        self.label_col = label_col

        self.schema = DataSet.Schema(self)
        return


    def drop(self, index :list[int|str]) -> 'DataSet':
        """
        Drops specified labels from rows.

        Parameters
        ----------
        index : list[int]
            Index labels to drop.

        Returns
        -------
        DataSet
            Returns a data set with the specified index labels removed.
        """
        df = self.data.drop(index=index, inplace=False)
        return DataSet(df, self.label_col)


    def get_feature_as_series(self, col :int|str) -> pd.Series:
        """
        Returns the value of the feature for each data point. 

        Parameters
        ----------
        col : int | str
            Name of the feature.

        Returns
        -------
        :pd.Series
            Series of feature values.

        Raises
        ------
        ValueError
            If ``col`` is not a column of the data set.
        """
        if col not in self.schema.features:
            raise ValueError(f"No feature named {col}")
        return self.data[col]


    def get_labels_as_series(self) -> pd.Series:
        """
        Returns the label of each data point.

        Returns
        -------
        :pd.Series
            Series of labels.

        Raises
        ------
        ValueError
            If the labels are not known for these data points.
        """
        if not self.schema.has_labels():
            raise ValueError("No labels in this dataset")
        return self.data[self.label_col]


    def sample(self, n :int|None=None, frac :float|None=None, replace :bool=True, seed :int=1) -> 'DataSet':
        """
        Returns a random sample.

        Parameters
        ----------
        n : int | None, optional
            Number of items to return, by default None. Cannot be used with ``frac``.
        frac : float | None, optional
            Fraction of items to return, by default None. Cannot be used with ``n``.
        replace : bool, optional
            Allow or disallow sampling of the same row more than once, by default True.
        seed : int | None, optional
            Seed for random number generator., by default 1.

        Returns
        -------
        DataSet
            A new object of same type containing ``n`` items randomly sampled from the caller object.
        """
        df = self.data.sample(n=n, frac=frac, replace=replace, random_state=seed)
        return DataSet(df, self.label_col)


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
        This inner class provides some useful methods for obtaining information regarding the structure of a data set.
        """
        def __init__(self, ds :'DataSet') -> None:
            """
            Parameters
            ----------
            ds : DataSet
                A data set.
            """
            self.ds = ds

            self.features = ds.data.columns.drop(self.ds.label_col) if self.ds.label_col is not None else ds.data.columns
            self.num_features = len(self.features)
            return


        def get_label_domain(self) -> np.ndarray:
            """
            Returns the domain of the label.

            Returns
            -------
            :np.ndarray
                An array of all the possible label values,
                so it contains unique values.

            Raises
            ------
            ValueError
                If the labels are not known for these data points.
            """
            return self.ds.get_labels_as_series().unique()


        def get_feature_domain(self, col :int|str) -> np.ndarray:
            """
            Returns the domain of the feature.


            Parameters
            ----------
            col : int | str
                Name of the feature.

            Returns
            -------
            :np.ndarray
                An array of all the possible values of the cosidered feature,
                so it contains unique values.
            
            Raises
            ------
            ValueError
                If ``col`` is not a column of the data set.
            """
            return self.ds.get_feature_as_series(col).unique()


        def get_type(self, col :int|str) -> DataType:
            """
            Returns the type of a feature.

            Parameters
            ----------
            col : int | str
                Name of the feature.

            Returns
            -------
            :DataType
                Type of the given feature.

            Raises
            ------
            ValueError
                If ``col`` is not a column of the data set.
            """
            if col not in self.features:
                raise ValueError(f"No feature named {col}")
            return DataType.CATEGORICAL if self.ds.data.loc[:, col].dtype ==  object else DataType.NUMERICAL
        

        def has_labels(self) -> bool:
            """
            Indicates whether the correct labels are reported in the data set.

            Returns
            -------
            :bool
                True iif expected labels are known.
            """
            return self.ds.label_col is not None