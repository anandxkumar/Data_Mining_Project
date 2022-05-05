# -*- coding: utf-8 -*-

### Categories id mapping to string


import glob, os
import json
import pandas as pd
from sklearn import preprocessing
import numpy as np
   
#### Creating a dictionary of category_id:category_name

def categories_id_to_name():
    def id_to_string():
           
        os.chdir("/home/pipebomb/Downloads/Data_Mining_Lab")
        list_json = []
        
        for file in glob.glob("*.json"):
            list_json.append(file)
        
           
        category_id = {}
        for json_file in list_json:
            with open(json_file) as f:
               data = json.loads(f.read())  
              
               for item in data['items']:
                    if item['id'] not in category_id:
                        category_id[item['id']] = item['snippet']['title']
            f.close()
        
        return category_id
    
    op = id_to_string()
    print(op)
    
    # Serializing json 
    json_object = json.dumps(op, indent = 4)
    
    with open("categories.json", "w") as outfile:
        outfile.write(json_object)
        
    outfile.close()



### Combining 3 csv files to form dataset

def combine():
    # importing pandas
    import pandas as pd
      
    # merging two csv files
    df = pd.concat(
        map(pd.read_csv, ['CAvideos.csv', 'GBvideos.csv', 'USvideos.csv']), ignore_index=True)
    print(df)
    
    df['ratio'] = df['likes']/df['dislikes']
    
    df.to_csv('final_dataset.csv', index=True)
    
'''
# normalize
a = pd.read_csv('CAvideos.csv')
b = pd.read_csv('USvideos.csv')
b['ratio'] = b['likes']/b['dislikes']
b = b.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
c = pd.read_csv('GBvideos.csv')
'''

# Loading and Cleaning the dataset
test = pd.read_csv('final_dataset.csv')
#test = test.dropna(subset=['ratio','views'])
# Replacing inf values by nan
test.replace([np.inf, -np.inf], np.nan, inplace=True)
# Removing all nan values
test.dropna(inplace=True)




#### Normalizing the fields

### Likes
x_array = np.array(test['likes'])
normalized_arr = preprocessing.normalize([x_array])
print(normalized_arr)
test['likes_n'] = normalized_arr[0].tolist()

### Dislikes
x_array = np.array(test['dislikes'])
normalized_arr = preprocessing.normalize([x_array])
print(normalized_arr)
test['dislikes_n'] = normalized_arr[0].tolist()

### Views
x_array_v = np.array(test['views'])
normalized_arr_v = preprocessing.normalize([x_array_v])
print(normalized_arr_v)
test['views_n'] = normalized_arr_v[0].tolist()

test['ratio_n'] = test['likes_n']/test['dislikes_n']


# Exporting final dataset
test.to_csv('final_dataset_2.csv', index=True)


#### Clustering usgin KMeans #####

test = pd.read_csv('final_dataset_2.csv')


from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

  

#### likes vs dislikes ####
t = test[['likes_n','dislikes_n']]
#t = t[t['ratio_n'] <= 0.06]
kmeans = KMeans(n_clusters=3).fit(t)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(t['likes_n'], t['dislikes_n'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)

plt.xlabel("Likes_normalized")
plt.ylabel("Dislikes_normalized")

plt.show()

from collections import Counter, defaultdict
print(Counter(kmeans.labels_))

####--------------------- constrained

t = test[['likes_n','dislikes_n']]

from k_means_constrained import KMeansConstrained
k = KMeansConstrained(n_clusters=3, size_min=100, size_max=None, init='k-means++', n_init=10, max_iter=100, tol=0.0001, verbose=False, random_state=None, copy_x=True, n_jobs=1).fit(t)


centroids = k.cluster_centers_
print(centroids)


plt.scatter(t['likes_n'], t['dislikes_n'], c= k.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)

plt.xlabel("Likes_normalized")
plt.ylabel("Dislikes_normalized")

plt.show()

from collections import Counter, defaultdict
print(Counter(k.labels_))

#### like vs views ####
t = test[['likes_n','views_n']]
#t = t[t['ratio_n'] <= 100]
kmeans = KMeans(n_clusters=3).fit(t)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(t['likes_n'], t['views_n'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)

plt.xlabel("likes_n")
plt.ylabel("views_n")

plt.show()

from collections import Counter, defaultdict
print(Counter(kmeans.labels_))


#### dislike vs views ####
t = test[['dislikes_n','views_n']]
#t = t[t['ratio_n'] <= 100]
kmeans = KMeans(n_clusters=3).fit(t)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(t['dislikes_n'], t['views_n'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)

plt.xlabel("dislikes_n")
plt.ylabel("views_n")

plt.show()

from collections import Counter, defaultdict
print(Counter(kmeans.labels_))




########## adding cluster_id to each row #####
count = {}

for i in test['cluster_id']:
    
    if i not in count:
        count[i] =1
    else:
        count[i] += 1

test.to_csv('output.csv')



category = {0:{},
            1:{},
            2:{}}

for ind in test.index:
    cluster = test['cluster_id'][ind]
    cat = int(test['category_id'][ind])
    if cat not in category[cluster]:
        category[cluster][cat] = 1
    else:
        category[cluster][cat] += 1

if __name__ == "__main__":
    op = categories_id_to_name()
    # Serializing json 
    json_object = json.dumps(op, indent = 4)
    # Writing to categories.json
    with open("categories.json", "w") as outfile:
        outfile.write(json_object)
    
    outfile.close()
    
    combine()