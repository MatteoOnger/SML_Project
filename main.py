import pandas as pd

from data import DataSet
from bintreepredictor import BinTreePredictor

train_ds = DataSet(pd.DataFrame([[1,2,"x"],[1,4,"x"], [5,6,"z"]], columns=["a", "b", "c"], index=["mark", "axel", "jude"]), "c")
print(train_ds)
print()

tree = BinTreePredictor("mode", "entropy", "max_nodes", 1)
tree.fit(train_ds)