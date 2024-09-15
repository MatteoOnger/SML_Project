import logging
import numpy as np
import pandas as pd

from data import DataSet
from bintreepredictor import BinTreePredictor


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


train_ds = DataSet(pd.DataFrame([[1,"1","x"],[2,"1","x"], [4.5,"1","z"],[5,"3","x"]], columns=["a", "b", "c"], index=["mark", "axel", "jude", "bob"]), "c")
print(train_ds)
print()

tree = BinTreePredictor("mode", "entropy", "max_nodes", 5, max_thresholds=2)
tree.fit(train_ds)

tree.print_tree()