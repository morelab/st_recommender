import csv
import json
import math
import statistics
import random
from collections import Counter

listOfStrat = ['v2', 'v5', 'v6', 'v7', 'v10', 'v11', 'v15', 'v17', 'v19', 'v20']

### AQUI SE CREA EL USERS FILE.
# jump = False
# with open('./data/raw/encuesta_GS_v3_country.csv', mode='r') as csv_file, open('./data/raw/Prolific.csv', mode='r') as csv_file_3:
#     csv_reader = csv.DictReader(csv_file)
#     line_count = 1
#     users = []
#     for row in csv_reader:
#         for i in listOfStrat:
#             if row[i] == '':
#                 jump = True
#         if jump:
#             jump = False
#             continue
#         user = {
#                 "userId": line_count,
#                 "features": {
#                     "0": None if row['Age'] == '' else int(row['Age']),
#                     "1": None if row['Gender'] == '' else int(row['Gender']),
#                     "2": None if row['Education'] == '' else int(row['Education']),
#                     "3": int(row['Country']),
#                     "4": None if row['Work_culture'] == '' else int(row['Work_culture']),
#                     "5": None if row['PST'] == '' else int(row['PST']),
#                     "6": None if row['Barriers'] == '' else int(row['Barriers']),
#                     "7": None if row['Intentions'] == '' else int(row['Intentions']),
#                     "8": None if row['Confidence'] == '' else int(row['Confidence'])
#                 }
#             }

#         users.append(user)
#         line_count += 1
#     print(line_count)

#     csv_reader = csv.DictReader(csv_file_3)
#     for row in csv_reader:

#         user = {
#                 "userId": line_count,
#                 "features": {
#                     "0": int(row['Age']),
#                     "1": int(row['Gender']),
#                     "2": None if row['Education'] == '' else int(row['Education']),
#                     "3": int(row['Country']),
#                     "4": int(row['Work_culture']),
#                     "5": None if row['PST'] == '' else int(row['PST']),
#                     "6": None if row['Barriers'] == '' else int(row['Barriers']),
#                     "7": None if row['Intentions'] == '' else int(row['Intentions']),
#                     "8": None if row['Confidence'] == '' else int(row['Confidence'])
#                 }
#             }

#         users.append(user)
#         line_count += 1
#     random.shuffle(users)

#     with open('users_pre_prolific_shuffle.json', 'w') as outfile:
#         json.dump(users, outfile)

# ids = ['v2', 'v5', 'v6', 'v7', 'v10', 'v11', 'v15', 'v17', 'v19', 'v20']
# with open('./data/processed/ratings.json') as json_file:
#     data = json.load(json_file)
#     ratings = []
#     for p in data['ratings']:
#         if p['itemId'] in ids:
#             ratings.append(p)
#     for i in range(302, 369):
#         for j in ids:
#             rating = {
#                 "userId": i,
#                 "itemId": j,
#                 "rating": None
#             }
#             ratings.append(rating)
#     with open('./data/processed/ratings_GS.json', 'w') as outfile:
#         json.dump(ratings, outfile)

# with open('./data/raw/GS_post.csv', mode='r') as csv_file, open('ratings.csv', 'w') as ratings_file:
#     csv_reader = csv.DictReader(csv_file)
#     line_count = 0
#     users = []
#     for row in csv_reader:
#         if line_count == 0:
#             # print(f'Column names are {", ".join(row)}')
#             line_count += 1
#         line_count += 1

#         ratings = {
#             'v2' : row['v2'],
#             'v5' : row['v5'],
#             'v6' : row['v6'],
#             'v7' : row['v7'],
#             'v10' : row['v10'],
#             'v11' : row['v11'],
#             'v15' : row['v15'],
#             'v17' : row['v17'],
#             'v19' : row['v19'],
#             'v20' : row['v20']
#         }
#         ratings = (sorted(ratings.items(), key = lambda kv:(int(kv[1]), kv[0])))
#         print(ratings[0])
#         exit()
#         ratings_writer = csv.writer(ratings_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

#         ratings_writer.writerow([301 + line_count, ratings[0][0],
#                                                     ratings[1][0],
#                                                     ratings[2][0],
#                                                     ratings[3][0],
#                                                     ratings[4][0],
#                                                     ratings[5][0],
#                                                     ratings[6][0],
#                                                     ratings[7][0],
#                                                     ratings[8][0],
#                                                     ratings[9][0]])

        

        # user = {
        #         "userId": 301 + line_count,
        #         "features": {
        #             "0": int(row['Age']),
        #             "1": int(row['Gender']),
        #             "2": None if row['Education'] == '' else int(row['Education']),
        #             "3": int(row['City']),
        #             "4": int(row['Work_culture']),
        #             "5": None if row['Profile_PST'] == '' else int(row['Profile_PST']),
        #             "6": None if row['Barriers'] == '' else int(row['Barriers']),
        #             "7": None if row['Intentions'] == '' else int(row['Intentions']),
        #             "8": None if row['Frequency'] == '' else int(row['Frequency'])
        #         }
        #     }

# with open('./data/processed/ratings.json') as json_file:
#     data = json.load(json_file)
#     user_id = set(range(368))
#     user_id2 = []
#     for i in data['ratings']:
#         user_id2.append(i['rating'])

#     print(statistics.mean(user_id2))
#     print(statistics.stdev(user_id2))


### 3 - AQUI SE CREA EL FILES DE RATINGS

# with open('./data/processed/rankings_all.csv', mode='r') as csv_file, open('./data/processed/ratings_all.json','w') as outfile:
#     csv_reader = csv.reader(csv_file)
#     line_count = 0
#     users = []
#     ratings = []
#     for row in csv_reader:
#         line_count += 1
#         for i in range(10):
#             rating = {
#                     "userId": line_count,
#                     "itemId": row[i],
#                     "rating": 4.61-(0.095*i)+random.uniform(0,0.19)

#                 }
#             ratings.append(rating)
    
#     json.dump(ratings, outfile)

# with open('./data/processed/ratings.json') as json_file:
#     data = json.load(json_file)
#     ratings = []
#     for p in data['ratings']:
#         if p['itemId'] in ids:
#             ratings.append(p)
#     for i in range(302, 369):
#         for j in ids:
#             rating = {
#                 "userId": i,
#                 "itemId": j,
#                 "rating": None
#             }
#             ratings.append(rating)
#     with open('./data/processed/ratings_GS.json', 'w') as outfile:
#         json.dump(ratings, outfile)


# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans


# with open('./data/processed/users_all.json') as json_file:
#     data = json.load(json_file)
#     new_train_x = [] 
#     for x in data:
#         f = []
#         for k, v in x['features'].items():
#             f.append(-1 if v is None or v == '' else v)
#         new_train_x.append(f)


# pca = PCA(n_components=2)
# x_r = pca.fit(new_train_x).transform(new_train_x)
# plt.figure()
# a1 = []
# a2 = []
# a5 = []
# a3 = []
# a4 = []
# b1 = []
# b2 = []

# for h in x_r:
#     a1.append(h[0])
#     a2.append(h[1])

# plt.scatter(a1, a2, c='#1f77b4')
# # plt.scatter(b1,b2,c='#ff7f0e')
# plt.show()


# kmeans = KMeans(n_clusters=4, random_state=0).fit(x_r)
# xx = kmeans.cluster_centers_
# for hh in xx:
#     a3.append(hh[0])
#     a4.append(hh[1])
# plt.scatter(a3, a4, color='r')
# plt.show()


# 2 - AQUI SE CREA EL RANKINGS FILE

# with open('./data/raw/encuesta_GS_v3_country.csv', 'r') as f_in, open('rankings_PRE.csv', 'w') as f_out:
#     csv_reader = csv.DictReader(f_in)
#     ratings = []
#     for row in csv_reader:
#         try:
#             res = {key: int(row[key]) for key in row.keys() 
#                                & listOfStrat}
#         except:
#             continue
#         good_dict = dict(sorted(res.items(), key=lambda item: item[1]))
#         # ratings.append(list(good_dict.keys()))
        
#         ratings_writer = csv.writer(f_out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         ratings_writer.writerow(list(good_dict.keys()))


# with open('./data/processed/ratings_GS.json', 'r') as f, open('rankings_all.csv', 'w') as f_out:
#     person_dict = json.load(f)
#     last_userId = None
#     d = {}
#     ratings = []
#     for row in person_dict:
#         if row['userId'] == last_userId or last_userId is None:
#             last_userId = row['userId']
#             d[row['itemId']] = row['rating']
#             print(d)         
#         else:
#             res = {key: int(d[key]) for key in d.keys() 
#                                & listOfStrat}
#             good_dict = dict(sorted(res.items(), key=lambda item: item[1]))
#             ratings.append(list(good_dict.keys()))
#             d = {}
#             userId = row['userId']
#             last_userId = row['userId']
#             d[row['itemId']] = row['rating']
    
#             ratings_writer = csv.writer(f_out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#             ratings_writer.writerow(list(good_dict.keys()))

# ranks = []
# with open('data/processed/rankings_all.csv', 'r') as csv_file:
#     csv_reader = csv.reader(csv_file)
#     for idx, row in enumerate(csv_reader, 1):
#         if idx <= 295 or idx >= 361:
#             for i in row[0:5]:
#                 ranks.append(i)

# print(Counter(ranks))


with open('users_pre_prolific_shuffle.json', 'r') as f:
    