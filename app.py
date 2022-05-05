from crypt import methods
from unicodedata import category
from flask import *
import pandas as pd

app = Flask(__name__)   

df =  pd.read_csv('op.csv')

@app.route('/home/<category>/', methods = ['GET','POST'])

def category_search(category):
    
    count = {'bar' : {}, 'table': {}}
    count['table'] = table_category(category)
    for ind, i in enumerate(df['cluster_id']):
        if df['category_name'][ind] == category:
            if i not in count['bar']:
                count['bar'][i] =1
            else:
                count['bar'][i] += 1

    return count


def table_category(category):
    rows = {
        0:[],
        1:[],
        2:[]
    }

    for ind in df.index:
        cluster = df['cluster_id'][ind]
        cat = df['category_name'][ind]
        if len(rows[cluster]) < 5 and cat == category:
            rows[cluster].append({'title': df['title'][ind], 'channel_title': df['channel_title'][ind], 
        'views': df['views'][ind], 'likes': df['likes'][ind], 'dislikes': df['dislikes'][ind]})

    return rows

@app.route('/', methods = ['GET','POST'])
def home():
    
    count = {'bar' : {}, 'pie' : {}, 'table': {}}
    count['table'] = table()
    count['pie'] = category_count()
    for i in df['cluster_id']:
        
        if i not in count['bar']:
            count['bar'][i] =1
        else:
            count['bar'][i] += 1

    return count

def category_count():

    category = {0:{},
                1:{},
                2:{}}

    for ind in df.index:
        cluster = df['cluster_id'][ind]
        cat = (df['category_name'][ind])
        if cat not in category[cluster]:
            category[cluster][cat] = 1
        else:
            category[cluster][cat] += 1

    return category


def table():
    rows = {
        0:[],
        1:[],
        2:[]
    }

    for ind in df.index:
        cluster = df['cluster_id'][ind]
        cat = df['category_name'][ind]
        if len(rows[cluster]) < 5:
            rows[cluster].append({'title': df['title'][ind], 'channel_title': df['channel_title'][ind], 
        'views': df['views'][ind], 'likes': df['likes'][ind], 'dislikes': df['dislikes'][ind]})

    return rows
            



if __name__ == '__main__':
    app.run(host = 'localhost', port = 5000)