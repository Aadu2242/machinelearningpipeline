import pandas as pd
from pymongo import MongoClient
import json
from arango import ArangoClient
from datetime import datetime

with open('/Users/adesh/PycharmProjects/machinelearningpipeline/src/templates/imdb_miner_template.json') as data_file:
    arguments = json.load(data_file)

df2 = pd.read_csv('/Users/adesh/PycharmProjects/machinelearningpipeline/src/data/imdb_master.csv',encoding="latin-1")

con = MongoClient(arguments['outputDS']["ip"], arguments['outputDS']["port"])
mongo_db = con['imdb']

train_data_to_upload = df2.to_dict('records')

mongo_db['train_imdb'].insert_many(train_data_to_upload)

df_test=pd.read_csv('/Users/adesh/PycharmProjects/machinelearningpipeline/src/data/testData.tsv',header=0, delimiter="\t", quoting=3)

test_data_to_upload = df_test.to_dict('records')

mongo_db = con['imdb']['test_imdb']

mongo_db.insert_many(test_data_to_upload)

arango_host = 'http://' + arguments['metricDS']["ip"] + ':' + arguments['metricDS']["port"]

client = ArangoClient(hosts=arango_host)

db = client.db(arguments['metricDS']["database"], username=arguments['metricDS']["username"], password=arguments['metricDS']["password"])
# Create a new collection named "students" if it does not exist.
# This returns an API wrapper for "students" collection.
arangodb = db.collection('output_status')

now = datetime. now()
dt_string = now. strftime("%d/%m/%Y %H:%M:%S")
output = {
		"Workflow" : arguments['name'],
		"execution_time" : dt_string,
        "status" : "Sucess"
	}

arangodb.insert(output)