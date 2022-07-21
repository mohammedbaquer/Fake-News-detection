from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

import nltk
nltk.download('stopwords')
nltk.download('punkt')
import nltk
nltk.download('wordnet')

stop_words = stopwords.words('english')
# stop_words = stopwords.words('english')


app = Flask(__name__)
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
loaded_model = pickle.load(open('model.pkl', 'rb'))
dataframe = pd.read_csv('news.csv')
x = dataframe['text']
y = dataframe['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lemmatizer=WordNetLemmatizer()
for index,row in dataframe.iterrows():
    filter_sentence = ''
    
    sentence = row['text']
    sentence = re.sub(r'[^\w\s]','',sentence) #cleaning
    
    words = nltk.word_tokenize(sentence) #tokenization
    
    words = [w for w in words if not w in stop_words]  #stopwords removal
    
    for word in words:
        filter_sentence = filter_sentence + ' ' + str(lemmatizer.lemmatize(word)).lower()
        
    dataframe.loc[index,'text'] = filter_sentence


def fake_news_det(news):
    tfid_x_train = tfvect.fit_transform(x_train)
    tfid_x_test = tfvect.transform(x_test)
    input_data = news

    # def remove_punctuation_stopwords_lemma(input_data):
    filter_sentence = ''
    lemmatizer=WordNetLemmatizer()
    input_data = re.sub(r'[^\w\s]','',input_data)
    # print(type(input_data))
    words = nltk.word_tokenize(input_data) #tokenization
    words = [w for w in words if not w in stop_words] 
    for word in words:
        filter_sentence = filter_sentence + ' ' + str(lemmatizer.lemmatize(word)).lower()
    # return filter_sentence
    input_data=[filter_sentence]
    print(input_data)

    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        # print(type(message))
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)