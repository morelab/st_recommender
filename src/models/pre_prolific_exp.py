import os
import sys
sys.path.insert(0, os.getcwd()+'/src/activelearning/')
sys.path.insert(0, os.getcwd()+'/src/data')
import json
import csv
import random
random.seed(1234)
import math
import pandas as pd
import numpy as np
np.random.seed(123)
import scipy.sparse as sp
# Baselines model function
from prepare_data import (get_users,
                          get_items,
                          get_ratings,
                          get_GS_users,
                          get_GS_ratings,
                          get_prolific_ratings,
                          get_Prolific_users,
                          get_all_ratings,
                          get_all_users,
                          get_GS_items)
from baselines_model import get_data
from lightfm import LightFM, data, datasets
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
from lightfm.cross_validation import random_train_test_split
from sklearn.ensemble import RandomForestRegressor
from collections import Counter

# listOfStrat = ['v2','v3','v4','v5','v6','v7','v8','v9','v10','v11','v12',
                # 'v13','v14','v15','v16','v17','v18','v19','v20','v21','v22']

listOfStrat = ['v2', 'v5', 'v6', 'v7', 'v10', 'v11', 'v15', 'v17', 'v19', 'v20']


def get_user_features():
    user_features = {'0',
                     '1',
                     '2',
                     '3',
                     '4',
                     '5',
                     '6',
                     '7',
                     '8'}
    return user_features


def get_item_features():
    item_features = {'0',
                     '1'}
    return item_features


def train_test_split(interactions, weights=None, test_percentage=0.2, i=0, users_len=0, shuffle=True, split=0):
    interactions = interactions.tocoo()
    shape = interactions.shape
    uids, iids, data = (interactions.row,
                        interactions.col,
                        interactions.data)
    shuffle = shuffle

    train_test_idx = list(range(users_len))

    if shuffle:
        random.shuffle(train_test_idx)

        train_perc = (int(len(train_test_idx) * 0.8))
        train_userid = train_test_idx[:train_perc]
    else:
        train_userid = train_test_idx[:296]
        test_userid = train_test_idx[360:]

    train_idx = []
    test_idx = []
    for i in range(len(uids)):
        if uids[i] in train_userid:
            train_idx.append(i)
        if uids[i] in test_userid:
            test_idx.append(i)

    train_interactions = sp.coo_matrix((data[train_idx],
                                       (uids[train_idx],
                                        iids[train_idx])),
                                        shape=shape,
                                        dtype=interactions.dtype)

    test_interactions = sp.coo_matrix((data[test_idx],
                                      (uids[test_idx],
                                      iids[test_idx])),
                                      shape=shape,
                                      dtype=interactions.dtype)
    

# WEIGHTS
    if weights is not None:
        weights = weights.tocoo()
        shape = weights.shape

        uids, iids, data = (weights.row,
                            weights.col,
                            weights.data)

        train_idx = []
        test_idx = []
        for i in range(len(uids)):
            if uids[i] in train_userid:
                train_idx.append(i)
            else:
                test_idx.append(i)

        train_weights = sp.coo_matrix((data[train_idx],
                           (uids[train_idx],
                            iids[train_idx])),
                          shape=shape,
                          dtype=weights.dtype)

        test_weights = sp.coo_matrix((data[test_idx],
                          (uids[test_idx],
                           iids[test_idx])),
                         shape=shape,
                         dtype=weights.dtype)

        return (train_interactions, test_interactions, 
                train_weights, test_weights, train_userid)

    return train_interactions, test_interactions, train_userid


def precision(prediction, truth, k=5, min=3):
    prediction = prediction.todense()
    truth = truth.todense()

    total_top = []
    top = []
    user_id = []
    for idx, i in enumerate(prediction):
        temp = np.squeeze(np.asarray(i))
        top = []
        if temp.any():
            user_id.append(idx)
            for idr, j in enumerate(temp):
                if j < k:
                    top.append(idr)
        total_top.append(top)
    
    #print(total_top)

    #Check the truth
    tot_precision = 0
    for idx, i in enumerate(truth):
        precision = 0
        if idx in user_id:
            temp = np.squeeze(np.asarray(i))
            #print(temp)
            for j in total_top[idx]:
                if temp[j] > min:
                    precision += 1
            tot_precision += precision/k
            
    return tot_precision/len(user_id)


def recall(prediction, truth, k=5, min=3):
    prediction = prediction.todense()
    truth = truth.todense()
    exceptions = 0

    total_top = []
    top = []
    user_id = []
    for idx, i in enumerate(prediction):
        temp = np.squeeze(np.asarray(i))
        top = []
        if temp.any():
            user_id.append(idx)
            for idr, j in enumerate(temp):
                if j < k:
                    top.append(idr)
        total_top.append(top)
    
    #print(total_top)

    #Check the truth
    tot_recall = 0
    for idx, i in enumerate(truth):
        recall = 0
        div_recall = 0
        if idx in user_id:
            temp = np.squeeze(np.asarray(i))
            #print(temp)
            for j in total_top[idx]:
                if temp[j] > min:
                    recall += 1
            for j in temp:
                if j > min:
                    div_recall += 1
            if recall == 0 and div_recall == 0:
                exceptions += 1
                continue
            tot_recall += recall/div_recall

    return tot_recall/(len(user_id) - exceptions)


def ndpm(predicted_ranks, truth, k=10):
    predicted_ranks = predicted_ranks.todense()
    predicted_total = []
    user_id = []

    for idx, i in enumerate(predicted_ranks):
        temp = np.squeeze(np.asarray(i))
        if temp.any() and len(np.unique(temp)) == k:
            temp_rank = []
            user_id.append(idx)
            # print(temp)
            for i in range(len(temp)):
                temp_rank.append(listOfStrat[np.where(temp == i)[0][0]])
            predicted_total.append(temp_rank[0:5])

    avg = 0
    beta = 0
    count = 0
    for idx, ix in enumerate(predicted_total):
        rnd = ix
        i = truth[user_id[idx]]
        # rnd = random.sample(listOfStrat, len(i))
        # rnd = ['v5', 'v6', 'v2', 'v11', 'v7']
        beta += len(set(i)-set(rnd))
        for j in i:
            # tengo v1
            for x in range(i.index(j)+1, len(i)):
                if j in rnd and i[x] in rnd:
                    # busco si v1 existe en el otro array y cojo la posicion
                    pos_i = i.index(j) - i.index(i[x])
                    # print(pos_i)
                    pos_rnd = rnd.index(j) - rnd.index(i[x])
                    # print(pos_rnd)
                    if (pos_i < 0 and pos_rnd) > 0 or (pos_i > 0 and pos_rnd < 0):
                        beta += 2
                    count += 1

        div = math.factorial(len(i))/(math.factorial(2)*math.factorial(len(i)-2))
        avg += float(beta)/(2*div)
        beta = 0
    return avg/len(predicted_total)



def get_model_data():
    #LightFM model

    json_user_features = get_user_features()
    json_item_features = get_item_features()
    
    dataset = data.Dataset(user_identity_features=True,
                              item_identity_features=False)

    dataset.fit(users=(x['userId'] for x in get_all_users()),
                items=(x['itemId'] for x in get_all_ratings()))

    for i in json_user_features:
        dataset.fit_partial(users=(x['userId'] for x in get_all_users()),
            user_features=(x['features'][i] for x in get_all_users()))


    for i in json_item_features:
        dataset.fit_partial(items=(x['itemId'] for x in get_GS_items()),
            item_features=(x['features'][i] for x in get_GS_items()))

    user_features = dataset.build_user_features(((x['userId'],
                                                [x['features']['0'],
                                                 x['features']['1'],
                                                 x['features']['2'],
                                                 x['features']['3'],
                                                 x['features']['4'],
                                                 x['features']['5'],
                                                 x['features']['6'],
                                                 x['features']['7'],
                                                 x['features']['8']
                                                 ]) for x in get_all_users()))

    item_features = dataset.build_item_features(((x['itemId'],
                                                [x['features']['0'],
                                                 x['features']['1']
                                                 ]) for x in get_GS_items()))

    (interactions, weights) = dataset.build_interactions(((x['userId'],
                                                           x['itemId'],
                                                           x['rating']) 
                                                           for x in get_all_ratings()))

    return interactions, weights, user_features


def recommender_benchmark(ranks, test_weights, total_top):
    # prec = precision(ranks, test_weights)
    # rec = recall(ranks,test_weights)
    # f = (2*prec*rec)/(prec+rec)
    prec = 0
    rec = 0
    f = 0
    ndp = ndpm(ranks, total_top)

    return prec, rec, f, ndp


if __name__ == '__main__':

    ndpms = []

    for i in range(5):
        total_truth = []
        # for l in total_top:
        #     total_truth.append(list(set(listOfStrat).intersection(set(l))))

        interactions, weights, user_features = get_model_data()

        (train_interactions, test_interactions,
        train_weights, test_weights, train_userid) = train_test_split(interactions,
                                                        weights,
                                                        test_percentage=0.2,
                                                        users_len=user_features.shape[0],
                                                        shuffle=False,
                                                        split=0)

        rankings_gs = []
        with open('./data/processed/rankings_all.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                total_truth.append(row)

        train_test_idx = set(range(user_features.shape[0]))
        test_ids = train_test_idx - set(train_userid)
        truth = [x for i, x in enumerate(total_truth) if i in test_ids]

        n = 0
        epochs = 100

        for epoch in range(epochs):
            model = LightFM()
            model.fit(train_interactions,
                    user_features=user_features,
                    sample_weight=train_weights,
                    epochs=epochs,
                    num_threads=4,
                    verbose=False)

                    

            ranks = model.predict_rank(test_interactions,
                                    train_interactions,
                                    user_features=user_features,
                                    check_intersections=True)

            n += ndpm(ranks, total_truth)
        file = open('ranks_pre_prolific.csv', 'w+', newline ='') 
        # writing the data into the file 
        with file:     
            write = csv.writer(file) 
            write.writerows(ranks.todense()) 
        
        ndpms.append(n/epochs)
        # prec, rec, f, ndp = recommender_benchmark(ranks,
        #                                           test_weights,
        #                                           rankings_gs)
    print('NDPM(Less is better): %f' % (np.mean(ndpms)))
        # print('Precision @ 5: %f' % (prec))
        # print('Recall @ 5: %f' % (rec))
        # print('F measure: %f' % (f))
        # print('NDPM(Less is better): %f' % (ndp))
