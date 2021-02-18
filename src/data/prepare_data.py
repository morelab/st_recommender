import os
import csv
import json
import numpy as np


def get_users():
    with open('data/processed/users_v_RF_v2.json') as f:
        users = json.load(f)
    return users['users']


def get_GS_users():
    with open('data/processed/GS_post.json') as f:
        users = json.load(f)
    return users['users']

def get_Prolific_users():
    with open('data/processed/users_prolific.json') as f:
        users = json.load(f)
    return users

def get_Prolific_post_users():
    with open('data/processed/users_prolific_post.json') as f:
        users = json.load(f)
    return users

def get_all_users():
    with open('data/processed/users_all.json') as f:
        users = json.load(f)
    return users

def get_items():
    with open('data/processed/items.json') as f:
        items = json.load(f)
    return items['items']


def get_GS_items():
    with open('data/processed/items_GS.json') as f:
        items = json.load(f)
    return items['items']


def get_ratings():
    with open('data/processed/ratings.json') as f:
        ratings = json.load(f)

    return ratings['ratings']


def get_GS_ratings():
    with open('data/processed/ratings_GS.json') as f:
        ratings = json.load(f)
    return ratings

def get_prolific_ratings():
    with open('data/processed/ratings_prolific.json') as f:
        ratings = json.load(f)
    return ratings

def get_prolific_post_ratings():
    with open('data/processed/ratings_prolific_post.json') as f:
        ratings = json.load(f)
    return ratings

def get_all_ratings():
    with open('data/processed/ratings_all.json') as f:
        ratings = json.load(f)
    return ratings

def get_features():
    users = get_users()
    
    all_feat = []
    feat = []
    for i in users:
        for j in i['features']:
            feat.append(i['features'][j])
        feat = [x if x != '' else -1 for x in feat]
        all_feat.append(feat)
        feat = []
    
    return np.asarray(all_feat)
    
def get_ratings_array():

    ROOT = os.path.abspath(os.path.dirname('sentient_things'))
    #print('root: %s' % (ROOT))
    DATA_LOCATION = os.path.join(ROOT, 'data/raw/encuesta_GS_v3.csv')
    #print('dataloc: %s' % (DATA_LOCATION))

    with open(DATA_LOCATION, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)

    dataset = np.genfromtxt(DATA_LOCATION, delimiter=',', 
            skip_header=1)


    ratings_tmp = []
    ratings = dataset[0:, [range(14,35)]]

    final = []
    for i in ratings:
        final.append(i[0])

    
    return np.asarray(final)