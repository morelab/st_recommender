import os
import sys
sys.path.insert(0, os.getcwd()+'/src/models/')
sys.path.insert(0, os.getcwd()+'/src/data/')

import random
random.seed(1234)
import numpy as np
np.random.seed(1234)
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean 
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.linear_model import LinearRegression
from sklearn import svm
# Custom imports
from prepare_data import (get_features, get_ratings_array)
from baselines_model import get_data

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.base import BaseEstimator
from modAL.batch import uncertainty_batch_sampling
import time

from sklearn.svm import LinearSVC
from modAL import ActiveLearner
from modAL.multilabel import (avg_score, avg_confidence, max_loss, max_score)
from modAL.uncertainty import (entropy_sampling,
                               margin_sampling, 
                               uncertainty_sampling)
from modAL.expected_error import expected_error_reduction
from sklearn.preprocessing import OneHotEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from modAL.density import information_density
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from sklearn.metrics import mean_absolute_error

def random_sampling(classifier, X_pool):
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples))
    return query_idx, X_pool[query_idx]

def density_sampling(classifier, X_pool):
    cosine_density = information_density(X_pool, 'euclidean')
    query_idx = [i for i,x in enumerate(cosine_density) if x == max(cosine_density)][0]
    return query_idx, X_pool[query_idx] 

def uncertainty_custom_sampling(regressor, X_pool):
    _, std = regressor.predict(X_pool, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X_pool[query_idx]

def split_train_test(dataset, user_features, train_ids=None, test_ids=None,
                     test_size=0.2):
    if(train_ids == None and test_ids == None):
        #Divide train and test users
        train_test_idx = list(range(0,len(dataset)))

        l = int(len(dataset) * (1-test_size))

        random.shuffle(train_test_idx)

        train_userid = train_test_idx[:l]
        test_userid = train_test_idx[l:]

        
    else:
        train_userid = train_ids
        test_userid = test_ids

    
    train_users = []
    test_users = []
    for idx, i in enumerate(dataset):
        if idx in train_userid:
            train_users.append(i)
        elif idx in test_userid:
            test_users.append(i)

    
    ################# -- END --- #################
    
    # Create matrix for train and test ratings
    # [item, 9xfeatures]
    ratings = [0,1,2,3,4,5]
    onehot_encoder = OneHotEncoder(sparse=False, categories=[ratings])
    train_rat_matrix = []
    test_rat_matrix = []
    train_feat_matrix = []
    test_feat_matrix = []

    strats = list(range(21))

    for idx, i in enumerate(train_users):
        locations = []
        if np.isnan(i).any():
            locations = np.argwhere(np.isnan(i))
            no_nan = np.delete(i, locations)
        else:
            no_nan = i
        locations = list(set(strats) - set(np.reshape(locations,len(locations))))
        onehot_encoded = onehot_encoder.fit_transform(no_nan.reshape(len(no_nan), 1))
        temp = []
        for idy, j in enumerate(no_nan):#enumerate(onehot_encoded):
            temp.append(locations[idy])
            for x in list(user_features[idx]):
                temp.append(x)
            train_feat_matrix.append(temp)
            train_rat_matrix.append(j)
            temp = []
    
    
            
    for idx, i in enumerate(test_users):
        locations = []
        if np.isnan(i).any():
            locations = np.argwhere(np.isnan(i))
            no_nan = np.delete(i, locations)
        else:
            no_nan = i
        locations = list(set(strats) - set(np.reshape(locations,len(locations))))
        onehot_encoded = onehot_encoder.fit_transform(no_nan.reshape(len(no_nan), 1))
        temp = []
        for idy, j in enumerate(no_nan):#enumerate(onehot_encoded):
            temp.append(locations[idy])
            for x in list(user_features[idx]):
                temp.append(x)
            test_feat_matrix.append(temp)
            test_rat_matrix.append(j)
            temp = []

    return (np.asarray(train_feat_matrix), 
            np.asarray(train_rat_matrix),  
            np.asarray(test_feat_matrix),
            np.asarray(test_rat_matrix))

def get_rank(dataset, user_features, train_ids, test_ids, ranks_test, ranks_train, num_ranks = 10):
    train_userid = train_ids
    test_userid = test_ids

    train_users = []
    test_users = []
    for idx, i in enumerate(dataset):
        if idx in train_userid:
            train_users.append(i)
        elif idx in test_userid:
            test_users.append(i)


    test_rat_matrix = []
    test_feat_matrix = []
    train_rat_matrix = []
    train_feat_matrix = []

    ranks_test = ranks_test.todense()
    ranks_train = ranks_train.todense()

    for idx, i in enumerate(train_users):
        rank = np.squeeze(np.asarray(ranks_train[train_userid[idx]]))[:num_ranks]
        temp = []
        for j in rank:
            if np.isnan(i[int(j)] ).any():
                continue
            temp.append(int(j))
            for x in list(user_features[train_userid[idx]]):
                temp.append(x)
            train_feat_matrix.append(temp)
            train_rat_matrix.append(i[int(j)])
            temp = []

    for idx, i in enumerate(test_users):
        rank = np.squeeze(np.asarray(ranks_test[test_userid[idx]]))[:num_ranks]
        temp = []
        for j in rank:
            if np.isnan(i[int(j)] ).any():
                continue
            temp.append(int(j))
            for x in list(user_features[test_userid[idx]]):
                temp.append(x)
            test_feat_matrix.append(temp)
            test_rat_matrix.append(i[int(j)])
            temp = []

   
    return (np.asarray(train_feat_matrix),
            np.asarray(train_rat_matrix),
            np.asarray(test_feat_matrix),
            np.asarray(test_rat_matrix))
                

def active_learning_models(train_x, train_y, neighbors):

    kernel = RBF(length_scale=7.0, length_scale_bounds=(1e-2, 1e3)) \
         + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
    
    gp = GaussianProcessRegressor(kernel=kernel, random_state=1234)
    learner_density = ActiveLearner(
            estimator=gp,
            query_strategy=density_sampling,
            X_training=train_x, y_training=train_y
        )

    learner_uncertainity = ActiveLearner(
            #estimator=KNeighborsClassifier(n_neighbors=neighbors),
            estimator=gp,
            query_strategy=uncertainty_custom_sampling,
            X_training=train_x, y_training=train_y
        )

    learner_random = ActiveLearner(
            #estimator=KNeighborsClassifier(n_neighbors=neighbors),
            estimator=gp,
            query_strategy=random_sampling,
            X_training=train_x, y_training=train_y
        )
    
    learner_density.fit(train_x, train_y)
    learner_uncertainity.fit(train_x, train_y)
    learner_random.fit(train_x, train_y)

    return learner_density, learner_uncertainity, learner_random

def active_learning_benchmark(model, X_pool, y_pool, test_x, test_y, queries=50, model_name=None):
    predic = []
    truth = []
    rmse = []
    acc = []

    total_x = []
    total_y = []
    total_x.append(X_pool[0])
    total_y.append(y_pool[0])
    for i in range(queries):
        query_idx, query_inst = model.query(X_pool)
        if model_name != 'Uncertainity':
            model.teach([X_pool[query_idx]], [y_pool[query_idx]])
            total_x.append(X_pool[query_idx])
            total_y.append(y_pool[query_idx])
        else:
            model.teach(X_pool[query_idx], y_pool[query_idx])
            total_x.append(X_pool[query_idx])
            total_y.append(y_pool[query_idx])
        (X_pool, y_pool) = (np.delete(X_pool, query_idx, axis=0), 
                            np.delete(y_pool, query_idx, axis=0)) 
        acc.append(model.score(test_x, test_y))
        predic = (model.predict(test_x))
        truth = (test_y)
        rmse.append(sqrt(mean_squared_error(truth, predic)))
        #rmse.append(mean_absolute_error(truth, predic))

    start = time.time()
    lr = LinearRegression().fit(total_x, total_y)
    lr_pr = lr.predict(test_x)
    end = time.time()
    print('LR: %f' % sqrt(mean_squared_error(test_y, lr_pr)))
    print('Time %f' % (end-start))

    start = time.time()
    svr = svm.SVR().fit(total_x, total_y)
    svr_pr = svr.predict(test_x)
    end = time.time()
    print('SVM: %f' % sqrt(mean_squared_error(test_y, svr_pr)))
    print('Time %f' % (end-start))
    return predic, truth, rmse, acc


if __name__ == '__main__':

    ratings, total_top, total_top5, ratings_dict = get_data()
    features = get_features()
    ratings = get_ratings_array()
    n_queries = 20
    for i in range(1):
        train_x, train_y, test_x, test_y = split_train_test(ratings, features)
        n_initial = int(len(train_x)/5)
        initial_idx = np.random.choice(range(len(train_x)), size=n_initial, replace=False)
        x_initial, y_initial = train_x[initial_idx], train_y[initial_idx]
        X_pool, y_pool = (np.delete(train_x, initial_idx, axis=0), 
                          np.delete(train_y, initial_idx, axis=0))

        
        (learner_density, 
            learner_uncertainity, 
            learner_random) = active_learning_models(x_initial, y_initial)
        
        predic, truth, rmse_den = active_learning_benchmark(learner_density,
                                                        X_pool,
                                                        y_pool,
                                                        test_x,
                                                        test_y,
                                                        n_queries)
        predic, truth, rmse_un = active_learning_benchmark(learner_uncertainity,
                                                        X_pool,
                                                        y_pool,
                                                        test_x,
                                                        test_y,
                                                        n_queries,
                                                        'Uncertainity')
        predic, truth, rmse_rnd = active_learning_benchmark(learner_density,
                                                        X_pool,
                                                        y_pool,
                                                        test_x,
                                                        test_y,
                                                        n_queries)                                                        
        
        """
        l_m_p, l_m_t, l_m_r = [], [], []
        l_u_p, l_u_t, l_u_r = [], [], []
        l_r_p, l_r_t, l_r_r = [], [], []


        margin_pool_x = X_pool
        margin_pool_y = y_pool
        uncertainity_pool_x = X_pool
        uncertainity_pool_y = y_pool
        random_pool_x = X_pool
        random_pool_y = y_pool
        
        for i in range(len(X_pool)):
            print('%i' % i)
            #Margin
            query_idx, query_inst = learner_margin.query(margin_pool_x)
            learner_margin.teach([margin_pool_x[query_idx]], [margin_pool_y[query_idx]])
            (margin_pool_x,
             margin_pool_y) = (np.delete(margin_pool_x, query_idx, axis=0), 
                                np.delete(margin_pool_y, query_idx, axis=0))      
            l_m_p.append(learner_margin.predict(test_x))
            l_m_t.append(test_y)
            l_m_r.append(sqrt(mean_squared_error(l_m_t, l_m_p)))
            

            #Uncertainity
            query_idx, query_inst = learner_uncertainity.query(uncertainity_pool_x)
            learner_uncertainity.teach(uncertainity_pool_x[query_idx], uncertainity_pool_y[query_idx])
            (uncertainity_pool_x,
             uncertainity_pool_y) = (np.delete(uncertainity_pool_x, query_idx, axis=0), 
                                     np.delete(uncertainity_pool_y, query_idx, axis=0))              
            l_u_p.append(learner_uncertainity.predict(test_x))
            l_u_t.append(test_y)
            l_u_r.append(sqrt(mean_squared_error(l_u_t, l_u_p)))

            #Random
            query_idx, query_inst = learner_random.query(random_pool_x)
            learner_random.teach([random_pool_x[query_idx]], [random_pool_y[query_idx]])
            (random_pool_x,
             random_pool_y) = (np.delete(random_pool_x, query_idx, axis=0), 
                                     np.delete(random_pool_y, query_idx, axis=0))              
            l_r_p.append(learner_random.predict(test_x))
            l_r_t.append(test_y)
            l_r_r.append(sqrt(mean_squared_error(l_r_t, l_r_p)))
        """

        df=pd.DataFrame({'x': range(len(rmse_den)), 
                         'y1': rmse_den, 
                         'y2': rmse_un,
                         'y3': rmse_rnd})
        df.to_csv('AL_total_test.csv')
        fig = plt.figure()
        plt.plot('x', 'y1', data=df,
                 color='tab:gray', 
                 linewidth=2, 
                 label='Density')

        plt.plot('x', 'y2', data=df, 
                 color='tab:red',
                 linewidth=2, 
                 label='Uncertainity')
        
        plt.plot('x', 'y3', data=df, 
                 color='tab:pink', 
                 linewidth=2, 
                 label='Random')
        #plt.plot('x', 'y3',data=df, marker= '.', markerfacecolor='tab:red', markersize=12, color='tab:red', linewidth=2)
        plt.legend()
        plt.show()
        
        fig.savefig("AL_total_test.png")

