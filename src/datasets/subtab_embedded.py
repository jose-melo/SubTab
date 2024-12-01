from pathlib import Path

import numpy as np
import pandas as pd
import os

from src.datasets.base import BaseDataset
from src.utils.models_utils import TASK_TYPE


map_dataset_to_task = {
    "helena_subtab": TASK_TYPE.MULTI_CLASS,
    "aloi_subtab": TASK_TYPE.MULTI_CLASS,
    "jannis_subtab": TASK_TYPE.MULTI_CLASS,
    "higgs_subtab": TASK_TYPE.BINARY_CLASS,
    "mnist_subtab": TASK_TYPE.MULTI_CLASS,
    "adult_subtab": TASK_TYPE.MULTI_CLASS,
    "california_subtab": TASK_TYPE.REGRESSION,
}


class SubTabEmbedded(BaseDataset):

    def __init__(self, args):
        super(SubTabEmbedded, self).__init__(args)

        self.args = args
        self.is_data_loaded = False
        self.task_type = map_dataset_to_task[args.data_set]
        self.dataset_name = self.args.data_set
        self.name = self.args.data_set

    def load(self):

        filepath = os.path.join(self.data_path, self.args.data_set)

        self.x_train = np.load(filepath + "/z_train.npy")
        self.y_train = np.load(filepath + "/y_train.npy")
        self.y = self.y_train
        self.x_test = np.load(filepath + "/z_test.npy")
        self.y_test = np.load(filepath + "/y_test.npy")

        self.N, self.D = self.x_train.shape
        # self.D -= 1

        self.cat_features = []
        self.num_features = list(range(self.D))

        self.is_data_loaded = True

        self.cardinalities = []
        self.num_or_cat = {}
        self.cls = 0
