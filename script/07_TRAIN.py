# %%
import pandas as pd
import numpy as np
import random
import scipy
import math
import os
# import seaborn as sns
import matplotlib.pyplot as plt
import talib
from tqdm import tqdm
import gc
import slackweb
import pickle
# from prophet import Prophet
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
from sklearnex import patch_sklearn
patch_sklearn()

# %%
os.chdir('/home/toshi/PROJECTS/PredStock')

# %%
from autogluon.tabular import TabularPredictor
from autogluon.core.dataset import TabularDataset
from autogluon.tabular.models.knn.knn_rapids_model import KNNRapidsModel

train_b = TabularDataset('train_cla.csv')

save_path = None
label_column = "RATE"
# metric = 'r2'
# metric = 'f1'
metric = "roc_auc"
# metric = 'log_loss'

gbm_options = [
    {"ag_args_fit": {"num_gpus": 1}, "num_leaves": 20, "ag_args": {"name_suffix": "A"}},
    {"ag_args_fit": {"num_gpus": 1}, "num_leaves": 40, "ag_args": {"name_suffix": "B"}},
    {"ag_args_fit": {"num_gpus": 1}, "num_leaves": 80, "ag_args": {"name_suffix": "C"}},
    {
        "ag_args_fit": {"num_gpus": 1},
        "num_leaves": 160,
        "ag_args": {"name_suffix": "D"},
    },
    {
        "ag_args_fit": {"num_gpus": 1},
        "extra_trees": True,
        "num_leaves": 20,
        "ag_args": {"name_suffix": "XTA"},
    },
    {
        "ag_args_fit": {"num_gpus": 1},
        "extra_trees": True,
        "num_leaves": 40,
        "ag_args": {"name_suffix": "XTB"},
    },
    {
        "ag_args_fit": {"num_gpus": 1},
        "extra_trees": True,
        "num_leaves": 80,
        "ag_args": {"name_suffix": "XTC"},
    },
    {
        "ag_args_fit": {"num_gpus": 1},
        "extra_trees": True,
        "num_leaves": 160,
        "ag_args": {"name_suffix": "XTD"},
    },
    "GBMLarge",
]

xgb_options = [
    {"ag_args_fit": {"num_gpus": 1}, "tree_method": "gpu_hist", "ag_args_ensemble": {"num_folds_parallel": 1},},
    {
        "max_depth": 5,
        "ag_args_fit": {"num_gpus": 1},
        "tree_method": "gpu_hist",
        "ag_args_ensemble": {"num_folds_parallel": 1},
        "ag_args": {"name_suffix": "A"},
    },
    {
        "max_depth": 4,
        "ag_args_fit": {"num_gpus": 1},
        "tree_method": "gpu_hist",
        "ag_args_ensemble": {"num_folds_parallel": 1},
        "ag_args": {"name_suffix": "B"},
    },
]

cat_options = [
    {"ag_args_fit": {"num_gpus": 1}},
    {"max_depth": 8, "ag_args_fit": {"num_gpus": 1}, "ag_args": {"name_suffix": "A"}},
    {"max_depth": 10, "ag_args_fit": {"num_gpus": 1}, "ag_args": {"name_suffix": "B"}},
]

xt_options = [
    {"n_estimators": 20, "n_jobs": 1, "ag_args": {"name_suffix": "A"}},
    {"n_estimators": 30, "n_jobs": 1, "ag_args": {"name_suffix": "B"}},
    {"n_estimators": 40, "n_jobs": 1, "ag_args": {"name_suffix": "C"}},
    {"n_estimators": 50, "n_jobs": 1, "ag_args": {"name_suffix": "D"}},
    {"n_estimators": 70, "n_jobs": 1, "ag_args": {"name_suffix": "E"}},
]

hyperparameters = {
    KNNRapidsModel: {
        "ag_args_fit": {"num_gpus": 1},
        "ag_args_ensemble": {"num_folds_parallel": 1},
    },
    "LR": {"ag_args_ensemble": {"num_folds_parallel": 1}},
    # "XGB": xgb_options,
    # 'CAT': cat_options,
    # 'GBM': gbm_options,
    # "XT": xt_options,
    'XGB': {'ag_args_fit': {'num_gpus': 1}, "ag_args_ensemble": {"num_folds_parallel": 1}},
    "CAT": {
        "ag_args_fit": {"num_gpus": 1},
        "ag_args_ensemble": {"num_folds_parallel": 1},
    },
    "GBM": [
        {
            "ag_args_fit": {"num_gpus": 1},
            "ag_args_ensemble": {"num_folds_parallel": 1},
            "extra_trees": True,
            "ag_args": {"name_suffix": "XT"},
        },
        {"ag_args_fit": {"num_gpus": 1}, "ag_args_ensemble": {"num_folds_parallel": 1}},
        # 'GBMLarge',
    ],
    # 'XT': {"ag_args_ensemble": {"num_folds_parallel": 1}},
    'NN_TORCH': {'ag_args_fit': {'num_gpus': 1}, "ag_args_ensemble": {"num_folds_parallel": 3}},
    # 'FASTAI': {'ag_args_fit': {'num_gpus': 1}, "ag_args_ensemble": {"num_folds_parallel": 2}},
    # 'TRANSF': {
    #     'ag_args_fit': {'num_gpus': 1},
    #     "ag_args_ensemble": {"num_folds_parallel": 1},
    #     "batch_size": 4096,
    #     "d_size": 16,
    #     },
}

hyperparameter_tune_kwargs = {
    "searcher": "auto",
    "scheduler": "local",
    "num_trials": 10,
}

new = TabularPredictor(
    label=label_column, eval_metric=metric, path=save_path, groups="group"
)

new.fit(
    train_data=train_b.drop(["Unnamed: 0"], axis=1),
    num_bag_folds=10,
    num_bag_sets=1,
    num_stack_levels=1,
    hyperparameters=hyperparameters,
    # hyperparameter_tune_kwargs = hyperparameter_tune_kwargs,
    save_space=True,
)

# %%
pickle.dump(new, open('ag_model_new.mdl', 'wb'))

# %%



