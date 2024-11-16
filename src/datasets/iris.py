from pathlib import Path

import numpy as np
import pandas as pd
import os
from sklearn.datasets import load_iris

from src.datasets.base import BaseDataset
from src.utils.models_utils import TASK_TYPE


class Iris(BaseDataset):

    def __init__(self, args):
        super(Iris, self).__init__(args)

        self.args = args
        self.is_data_loaded = False
        self.name = "iris"
        self.task_type = TASK_TYPE.BINARY_CLASS

    def load(self):

        data = load_iris()

        mask = data.target < 2
        self.y = data.target[mask]
        self.X = data.data[mask]

        self.N, self.D = self.X.shape

        self.cat_features = []
        self.num_features = list(range(self.D))

        self.is_data_loaded = True

        self.cardinalities = []
        self.num_or_cat = {}
