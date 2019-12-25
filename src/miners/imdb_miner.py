import pandas as pd
from pymongo import MongoClient
import json

with open('src/templates/imdb_miner_template.json') as data_file:
    arguments = json.load(data_file)

df2 = pd.read_csv('src/data/imdb_master.csv',encoding="latin-1")
df2.head()

print(arguments)

con = MongoClient(arguments['outputDS']["ip"], arguments['outputDS']["port"])
mongo_db = con['imdb']

train_data_to_upload = df2.to_dict('records')

mongo_db['train_imdb'].insert_many(train_data_to_upload)

df_test=pd.read_csv('src/data/testData.tsv',header=0, delimiter="\t", quoting=3)

test_data_to_upload = df_test.to_dict('records')

mongo_db = con['imdb']['test_imdb']

mongo_db.insert_many(test_data_to_upload)