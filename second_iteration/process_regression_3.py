from data_prep import *
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.tree import export_graphviz
import pydot

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%matplotlib qt
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing, metrics
from sklearn.decomposition import PCA, KernelPCA
from sklearn import datasets
from sklearn.preprocessing import scale
from itertools import product
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

import pickle

# load data and variables of interest
df_pre = prepare_data_pre()
df_post = prepare_data_post()
predictor_vars_full, predictor_vars_ltd, target_vars = get_predictors_targets()
df_pre_full = df_pre[predictor_vars_full + target_vars].dropna()
df_pre_ltd = df_pre[predictor_vars_ltd + target_vars].dropna()
df_post = df_post[predictor_vars_ltd + target_vars].dropna()

df_pre_full.to_csv("df_pre_full.csv", index = False)
df_pre_ltd.to_csv("df_pre_ltd.csv", index = False)
df_post.to_csv("df_post.csv", index = False)

# create pairs of features and labels
X_pre_full = np.array(df_pre_full[predictor_vars_full])
Y_pre_full = np.array(df_pre_full[target_vars])
X_pre_ltd = np.array(df_pre_ltd[predictor_vars_ltd])
Y_pre_ltd = np.array(df_pre_ltd[target_vars])
X_post = np.array(df_post[predictor_vars_ltd])
Y_post = np.array(df_post[target_vars])

# remove all records that have a target value that appears only once in the target variable
def remove_singular_records(X, Y):
    has_more = True
    while has_more:
        all_single_values = []
        for i in range(Y.shape[1]):
            y = Y[:, i]
            vals, counts = np.unique(y, return_counts = True)
            single_values = vals[counts == 1]
            all_single_values += list(single_values)
            X = X[~np.isin(y, single_values), :]
            Y = Y[~np.isin(y, single_values), :]
            # print(np.unique(Y[:, i], return_counts=True))
        print(all_single_values)
        if len(all_single_values) <= 0:
            has_more = False
    return X, Y

# duplicate all records that have a target value that appears only once in the target variable
def duplicate_singular_records(X, Y):
    has_more = True
    while has_more:
        all_single_values = []
        for i in range(Y.shape[1]):
            y = Y[:, i]
            vals, counts = np.unique(y, return_counts = True)
            single_values = vals[counts == 1]
            all_single_values += list(single_values)
            
            X = np.concatenate((X, X[np.isin(y, single_values), :]))
            Y = np.concatenate((Y, Y[np.isin(y, single_values), :]))
            
#            X = X[~np.isin(y, single_values), :]
#            Y = Y[~np.isin(y, single_values), :]
#            # print(np.unique(Y[:, i], return_counts=True))
        print(all_single_values)
        if len(all_single_values) <= 0:
            has_more = False
    return X, Y

X_pre_full, Y_pre_full = remove_singular_records(X_pre_full, Y_pre_full)
X_pre_ltd, Y_pre_ltd = remove_singular_records(X_pre_ltd, Y_pre_ltd)
print(X_post.shape, Y_post.shape)
X_post, Y_post = remove_singular_records(X_post, Y_post)
print("AAAAAAAAAAAAA")
print(X_post.shape, Y_post.shape)

# scale post-pilot targets to 1-6 range instead of 1-10 range
Y_post = (Y_post - 1) / 9 * 5 + 1


def model_selection(X, y):
    model = RandomForestRegressor()
    # model = SVR()
    hyper_params = [ {'n_estimators': [100]}]#, 200, 300, 400, 500, 600]}]
    # hyper_params = [ {'kernel': ['rbf']}]
    folds = KFold(n_splits = 5)
    model_cv = GridSearchCV(estimator=model, 
                            param_grid=hyper_params, 
                            scoring='neg_mean_squared_error', 
                            cv=folds, 
                            verbose=4,
                            n_jobs=-1,
                            return_train_score=True,
                            refit=True)
    model_cv.fit(X, y)
    
    best_score = model_cv.best_score_
    best_hyperparams = model_cv.best_params_
    print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))
    
    return model_cv

def regression(X_train, X_test, y_train, y_test, do_pca):
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    if do_pca:
        pca = PCA(.9)
        # pca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
        pca.fit(X_train)
        print("PCA components", pca.n_components_)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
    
    model_cv = model_selection(X_train, y_train)
    
    y_pred = model_cv.best_estimator_.predict(X_test)
    
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    
    return model_cv, scaler, y_pred, rmse, model_cv.best_params_


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def print_results(X_train, X_test, Y_train, Y_test):
    results = {}
    Y_pred = np.zeros(Y_test.shape)
    models = []
    scalers = []
    for i in range(len(target_vars)):
        results[target_vars[i]] = {}
        for do_pca in [False]:#, True]:
            pca_attr = 'with_pca' if do_pca else 'no_pca'
            # X_train, X_test, y_train, y_test = train_test_split(X_pre_full, Y_pre_full[:, i], test_size = 0.25, stratify = Y_pre_full[:, 0], random_state = 42)
            model_cv, scaler, Y_pred[:, i], rmse, best_params = regression(X_train, X_test, Y_train[:, i], Y_test[:, i], do_pca)
            models.append(model_cv)
            scalers.append(scaler)
            results[target_vars[i]][pca_attr] = rmse
    print("models", models)
    params = {
        "models": models,
        "scalers": scalers
    }
    pickle.dump(params, open("model_params.p", "wb"))

    # print("Y_pred", Y_pred)
    print("Y_test", np.round(Y_test[1:10, :], 2))
    print("Y_pred", np.round(Y_pred[1:10, :], 2))
    c_test = np.argmax(Y_test, axis = 1)
    c_pred = np.argmax(Y_pred, axis = 1)
    print(c_test[1:10])
    print(c_pred[1:10])
    f1 = metrics.f1_score(c_test, c_pred, average='macro')
    print("f1 score", f1)
    
    C_test = list(map(lambda x : [i for i in range(len(x)) if x[i] == max(x)], Y_test))
    C_test_binary = np.zeros(Y_test.shape)
    for i in range(Y_test.shape[0]):
        C_test_binary[i, C_test[i]] = 1
    print("C_test", C_test[1:10])
    print("C_test_binary", C_test_binary[1:10])
    C_pred = list(map(lambda x : [i for i in range(len(x)) if x[i] == max(x)], Y_pred))
    C_pred_binary = np.zeros(Y_pred.shape)
    for i in range(Y_pred.shape[0]):
        C_pred_binary[i, C_pred[i]] = 1
    print("C_pred", C_pred[1:10])
    print("C_pred_binary", C_pred_binary[1:10])
    acc = sum([int(c_pred[i] in C_test[i]) for i in range(len(c_pred))]) / len(c_pred)
    print("Accuracy", acc)    
    hamming = hamming_score(C_test_binary, C_pred_binary)
    print("Hamming score", hamming)
    
    sum_acc = 0
    n_classes = 0
    for i in range(len(target_vars)):
        c_pred_filtered = [x for x in c_pred if x == i]
        if (len(c_pred_filtered) <= 0):
            continue
        n_classes += 1
        C_test_filtered = [C_test[j] for j in range(len(c_pred)) if c_pred[j] == i]
        print("Class", i)
        acc_filtered = sum([int(c_pred_filtered[i] in C_test_filtered[i]) for i in range(len(c_pred_filtered))]) / len(c_pred_filtered)
        print("Accuracy per class", acc_filtered)
        sum_acc += acc_filtered
    avg_acc = sum_acc / n_classes
    print("Average accuracy per class", avg_acc)
    
    # Ranking metrics
    
    
    resultsDf = pd.DataFrame.from_dict(results)
    print(resultsDf)
    return resultsDf


def plot_results(resultsDf):
    n_groups = resultsDf.shape[1]
    
    fig, ax = plt.subplots()
    
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.4
    
    plt.bar(index, list(resultsDf.loc['no_pca']), bar_width,
                     alpha=opacity,
                     color='b',
                     label='No PCA')
    
    plt.bar(index + bar_width, list(resultsDf.loc['with_pca']), bar_width,
                     alpha=opacity,
                     color='r',
                     label='With PCA')
    
    plt.xlabel('Strategy')
    plt.ylabel('RMSE')
    plt.title('RMSE per strategy')
    plt.xticks(index + bar_width / 2, list(resultsDf.columns))
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_results_no_pca(resultsDf):
    n_groups = resultsDf.shape[1]
    
    fig, ax = plt.subplots()
    
    index = np.arange(n_groups)
    opacity = 0.4
    
    plt.bar(index, list(resultsDf.loc['no_pca']), 
                     alpha=opacity,
                     color='b')
    
    plt.xlabel('Persuasion principle')
    plt.ylabel('RMSE')
    plt.title('RMSE per persuasion principle')
    plt.xticks(index, list(resultsDf.columns), rotation = 45, ha = 'right')
    
    plt.tight_layout()
    plt.show()


## Training and testing with pre-pilot
#X_train, X_test, Y_train, Y_test = train_test_split(X_pre_full, Y_pre_full, test_size = 0.25, stratify = Y_pre_full[:, 0], random_state = 42)

## Training and testing with pre-pilot (limited attributes)
#X_train, X_test, Y_train, Y_test = train_test_split(X_pre_ltd, Y_pre_ltd, test_size = 0.25, stratify = Y_pre_ltd[:, 0], random_state = 42)

## Training and testing with post-pilot
#X_train, X_test, Y_train, Y_test = train_test_split(X_post, Y_post, test_size = 0.25, stratify = Y_post[:, 0], random_state = 42)


## Training with pre-pilot and testing with post-pilot
#X_train, _, Y_train, _ = train_test_split(X_pre_ltd, Y_pre_ltd, test_size = 0.25, stratify = Y_pre_ltd[:, 0], random_state = 42)
#X_test = X_post
#Y_test = Y_post

## Training and testing with concatenated dataset
#X_concat = np.concatenate((X_pre_ltd, X_post))
#Y_concat = np.concatenate((Y_pre_ltd, Y_post))
#X_train, X_test, Y_train, Y_test = train_test_split(X_concat, Y_concat, test_size = 0.25, stratify = Y_concat[:, 0], random_state = 42)

# Training and testing with concatenated dataset (test on city 6 and 8)
X_concat = np.concatenate((X_pre_ltd, X_post))
Y_concat = np.concatenate((Y_pre_ltd, Y_post))
cityIdx = predictor_vars_ltd.index("city")
testCities = [6, 8]
X_train = X_concat[~np.isin(X_concat[:, cityIdx], testCities), :]
Y_train = Y_concat[~np.isin(X_concat[:, cityIdx], testCities), :]
X_test = X_concat[np.isin(X_concat[:, cityIdx], testCities), :]
Y_test = Y_concat[np.isin(X_concat[:, cityIdx], testCities), :]


resultsDf = print_results(X_train, X_test, Y_train, Y_test)
# plot_results(resultsDf)
plot_results_no_pca(resultsDf)
