import os
import sys
import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import db_connection
import matplotlib.pyplot as plt

# Импортируем библиотеки
from sklearn import datasets
from sklearn.cluster import KMeans


db = db_connection.PostgreConnection()

conn, cur = db.connection()

cur.execute('SELECT * FROM SENTIMENT_DATA')
rows = cur.fetchall()

user = [x[2] for x in rows]
sentiment = [[x[3]] for x in rows]

model = KMeans(n_clusters=4)

model.fit(sentiment)


all_predictions = model.predict(sentiment)

info = list(zip(all_predictions, user, sentiment))

sorted_list = sorted(info)

clas = [val[0] for val in sorted_list]
set_class = list(set(clas))
vall = [val[2] for val in sorted_list] 

categories = np.unique(clas)
colors = [plt.cm.tab10(i/float(len(categories)-1)) for i in range(len(categories))]

plt.xlabel("class")
plt.ylabel("value")
for ind, j in enumerate(set_class):
    cl = []
    val = []
    for index, category in enumerate(clas):
        if clas[index] == j:
            cl.append(clas[index])
            val.append(vall[index])

    plt.scatter(cl, val, s=20, c=colors[j], label=str(category))
    centers = model.cluster_centers_
    plt.scatter(j,centers[ind], c='black', s=25, alpha=0.7)
    

centers = model.cluster_centers_
print(centers)
plt.show()