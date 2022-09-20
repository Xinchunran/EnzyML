import random
from collections import Counter
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import esm
import scipy
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.pipeline import Pipeline
import xgboost
from xgboost import XGBRegressor
import argparse
import os

def log_trans(xs_data: float): 
    return np.log(xs_data)

def get_args():
    """the args for the algroithm"""
    desc = "esm path and esm alignment files"
    try:
        parser = argparse.ArgumentParser(
            description=desc, formatter_class=argparse.RawTextHelpFormatter
        )
        parser.add_argument(
            "Fasta_path",
            acction = "store",
            help = "File of original sequence files",
        )
        parser.add_argument(
            "Embedding_path",
            action="store",
            help = "File of embedding sequence files",
        )
        parser.add_argument(
            "Values",
            action="store",
            help = "File of values should do regression"
        )
        if len(sys.argv) == 1:
            parser.print_help(sys.stderr)
            exit(1)
    except:
        sys.stderr.write("An exception occurred with argument parsing. Check your provided options. \n")

    return parser.parse_args()

def data_loader(fasta_path:str, embd_path:str, value_path:str):
    ys, xs = [], []
    for header, _seq in esm.data.read_fasta(fasta_path):
        fn = f'{embd_path}/{header[1:]}.pt'
        embs = torch.load(fn)
        xs.append(embs['mean_representations'][EMB_LAYER])

    xs = torch.stack(xs, dim=0).numpy()

    kcat_wo_5 = pd.read_csv(value_path) #load the value data for the regression task
    value = kcat_wo_5.to_numpy()
    for i in value: 
        ys.append(float(i))

    #for i in value:
    #    ys.append(log_trans(float(i)))

    return xs, ys

def pca_analysis(xs: float, componets: int):
    pca = PCA(componets)
    xs_pca =  pca.fit_transform(xs)
    return xs_pca

def fig_pca(xs_data: float, ys_data: float):
    fig_dims = (7, 6)
    fig, ax = plt.subplots(figsize=fig_dims)
    sc = ax.scatter(xs_data[:,0], xs_data[:,1], c=ys_data, marker='.')
    ax.set_xlabel('PCA first principal component')
    ax.set_ylabel('PCA second principal component')
    plt.colorbar(sc, label='two components')
    plt.savefig("./pca_analysis.pdf", format="pdf")

    return print("analyzing the pca")

#knn regressor setting
knn_grid = [
    {
            'model': [KNeighborsRegressor()],
            'model__n_neighbors': [5, 10],
            'model__weights': ['uniform', 'distance'],
            'model__algorithm': ['ball_tree', 'kd_tree', 'brute'],
            'model__leaf_size' : [15, 30],
            'model__p' : [1, 2],
        }
    ]

#svm regressor setting
svm_grid = [
    {
            'model': [SVR()],
            'model__C' : [0.1, 1.0, 10.0],
            'model__kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
            'model__degree' : [3],
            'model__gamma': ['scale'],
        }
]

#random forest regressor setting
rfr_grid = [
    {
            'model': [RandomForestRegressor()],
            'model__n_estimators' : [20],
            'model__criterion' : ['squared_error', 'absolute_error'],
            'model__max_features': ['sqrt', 'log2'],
            'model__min_samples_split' : [5, 10],
            'model__min_samples_leaf': [1, 4]
        }
]

#xgboost regressor setting
xgb_grid = [
    {
            'model': [XGBRegressor()],
            'model__n_estimators' : [100],
            'model__criterion': ['squared_error', 'absolute_error'],
            'model__max_depth': [20], 
            'model__eta': [0.1],
            'model__subsample': [0.7], 
            'model__colsample_bytree': [0.8],
        }
]


def main():
    args = get_args()
    fasta_pth = args.Fasta_path
    embed_pth = args.Embedding_path
    value_pth = args.Values
    xs, ys = data_loader(fasta_pth, embed_pth, value_pth)
    train_size = 0.8
    xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, train_size=train_size, random_state=42)
    #xs_train.shape, xs_test.shape, len(ys_train), len(ys_test)
    num_pca_components = 60
    xs_train_pca = pca_analysis(xs_train, num_pca_components)
    fig_pca(xs_train_pca, ys_train) #print the pca analysis graph
    cls_list = [KNeighborsRegressor, SVR, RandomForestRegressor, XGBRegressor]
    param_grid_list = [knn_grid, svm_grid, rfr_grid, xgb_grid]
    pipe = Pipeline(
        steps = (
            ('pca', PCA(num_pca_components)),
            ('model', 'passthrough')
        )
    )

    result_list = []
    grid_list = []
    for cls_name, param_grid in zip(cls_list, param_grid_list):
        print(cls_name)
        grid = GridSearchCV(
            estimator = pipe,
            param_grid = param_grid,
            scoring = 'r2',
            verbose = 1,
            n_jobs = -1 # use all available cores
        )
        grid.fit(xs_train, ys_train)
        result_list.append(pd.DataFrame.from_dict(grid.cv_results_))
        grid_list.append(grid)

    for grid in grid_list:
        print(grid.best_estimator_.get_params()["steps"][1][1]) # get the model details from the estimator
        print()
        preds = grid.predict(xs_test)
        print(f'{scipy.stats.spearmanr(ys_test, preds)}')
        print('\n', '-' * 80, '\n')



