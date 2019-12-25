from arango import ArangoClient
import pandas as pd
from pymongo import MongoClient
import json
import re
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers


nltk.download('stopwords')
nltk.download('wordnet')


with open('/Users/adesh/PycharmProjects/machinelearningpipeline/src/templates/imdb_advisor_template.json') as data_file:
    arguments = json.load(data_file)


con = MongoClient(arguments['inputDS']["ip"], arguments['inputDS']["port"])
mongo_db = con['imdb']

cursor = mongo_db['train_imdb'].find()

df = pd.DataFrame(list(cursor))

df = df.drop(['Unnamed: 0','type','file', '_id'],axis=1)
df.columns = ["review","sentiment"]
df.head()

df = df[df.sentiment != 'unsup']
df['sentiment'] = df['sentiment'].map({'pos': 1, 'neg': 0})
df.head()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

df['Processed_Reviews'] = df.review.apply(lambda x: clean_text(x))


df.Processed_Reviews.apply(lambda x: len(x.split(" "))).mean()

max_features = 6000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df['Processed_Reviews'])
list_tokenized_train = tokenizer.texts_to_sequences(df['Processed_Reviews'])

maxlen = 130
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
y = df['sentiment']


embed_size = 128
model = Sequential()
model.add(Embedding(max_features, embed_size))
model.add(Bidirectional(LSTM(32, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 1000
epochs = 1
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

cursor = mongo_db['test_imdb'].find()

df_test = pd.DataFrame(list(cursor))
df_test.head()
df_test = df_test.drop(['_id'],axis=1)
df_test["review"]=df_test.review.apply(lambda x: clean_text(x))
df_test["sentiment"] = df_test["id"].map(lambda x: 1 if int(x.strip('"').split("_")[1]) >= 5 else 0)
y_test = df_test["sentiment"]
list_sentences_test = df_test["review"]
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
prediction = model.predict(X_te)
y_pred = (prediction > 0.5)

print (prediction)

print("Hello")

print(y_test)

arango_host = 'http://' + arguments['outputDS']["ip"] + ':' + arguments['outputDS']["ip"]

client = ArangoClient(hosts=arango_host)

db = client.db(arguments['outputDS']["database"], username=arguments['outputDS']["username"], password=arguments['outputDS']["password"])
# Create a new collection named "students" if it does not exist.
# This returns an API wrapper for "students" collection.
students = db.collection('predictions')

prediction= []

for x in np.nditer(y_pred):
    x=x.tostring()
    prediction.append(x)

submission = pd.DataFrame({
        "PassengerId": df_test['id'],
        "Survived": prediction
    })

prediction_data_to_upload = submission.T.to_dict().values()

students.insert_many(prediction_data_to_upload)

from sklearn.metrics import f1_score, confusion_matrix
print('F1-score: {0}'.format(f1_score(y_pred, y_test)))
print('Confusion matrix:')
confusion_matrix(y_pred, y_test)
