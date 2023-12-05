import streamlit as sl
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def txt_preprocess(msg):
    
    # convert all to lowercase
    msg = msg.lower()
    msg = nltk.word_tokenize(msg)
    
    # remove special characters 
    temp = []
    for i in msg:
        if i.isalnum():
            temp.append(i)
    msg = temp[:]
    temp.clear()
    
    # remove stopwords and punctuations
    for i in msg:
        if i not in stopwords.words('english') and i not in string.punctuation:
            temp.append(i) 
    msg = temp[:]
    temp.clear()
    
    # stemming - keeping only the root words (danc = dance = dancing = dances)
    for i in msg:
        temp.append(ps.stem(i))

    return " ".join(temp)
    
model = pickle.load(open('mnb_model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

sl.title("SMS / E-mail Spam Classifier")
sl.write("This prototype can detect whether a given textual message is spam or not")
input = sl.text_area("Enter the message / Email...")

if sl.button("Predict"):
    
    # Data preprocess
    processed_text = txt_preprocess(input)
    
    # Vectorize data
    vector_input = tfidf.transform([processed_text])
    
    # Predict
    result = model.predict(vector_input)[0]
    
    # Display the result
    if result == 1:
        sl.header("SPAM")
    else:
        sl.header("NOT A SPAM")
                                   

