from flask import Flask, render_template, request, jsonify
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import json
import random
from keras.models import load_model

lemmatizer = WordNetLemmatizer()

app = Flask(__name__)

# Load model and data
model = load_model('chatbot_Application_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
labels = pickle.load(open('labels.pkl', 'rb'))

def bag_of_words(s, words):
    bag = [0] * len(words)
    sentence_words = nltk.word_tokenize(s)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

    for sw in sentence_words:
        for i, w in enumerate(words):
            if w == sw:
                bag[i] = 1

    return np.array(bag)

def predict_label(s):
    bow = bag_of_words(s, words)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.60

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    if len(results) == 0:
        return [{"intent": "noanswer"}]

    return [{"intent": labels[r[0]]} for r in results]

def get_response(ints):
    tag = ints[0]['intent']
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

    return "Sorry, I didn't understand."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_text = request.json["message"]
    ints = predict_label(user_text)
    response = get_response(ints)
    return jsonify({"reply": response})

if __name__ == "__main__":
    app.run(debug=True)
