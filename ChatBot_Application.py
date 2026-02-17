import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import json
import random
from keras.models import load_model

model = load_model('chatbot_Application_model.h5')

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
labels = pickle.load(open('labels.pkl','rb'))


def bank_of_words(s,words, show_details=True):
    bag_of_words = [0 for _ in range(len(words))]
    sent_words = nltk.word_tokenize(s)
    sent_words = [lemmatizer.lemmatize(word.lower()) for word in sent_words]
    for sent in sent_words:
        for i,w in enumerate(words):
            if w == sent:
                bag_of_words[i] = 1
    return np.array(bag_of_words)

def predict_label(s, model):
    # filtering out predictions
    pred = bank_of_words(s, words,show_details=False)
    response = model.predict(np.array([pred]))[0]
    ERROR_THRESHOLD = 0.25
    final_results = [[i,r] for i,r in enumerate(response) if r>ERROR_THRESHOLD]
    final_results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in final_results:
        return_list.append({"intent": labels[r[0]], "probability": str(r[1])})
    return return_list

def Response(ints, intents_json):
    tags = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tags):
            response = random.choice(i['responses'])
            break
    return response

def chatbot_response(msg):
    ints = predict_label(msg, model)
    response = Response(ints, intents)
    return response
            
# def chat():
#     print("ARC Bot")
#     while True:
#         inp = input("You: ")
#         if inp.lower() == 'quit':
#             break
#         response = chatbot_response(inp)
#         print("\n BOT: " + response + '\n\n')

def chat():
    print("ARC Bot")
    
    while True:
        inp = input("You: ")

        ints = predict_label(inp, model)

        if len(ints) == 0:
            print("\n BOT: I didn't understand that.\n")
            continue

        response = Response(ints, intents)

        print("\n BOT: " + response + "\n")

        # 🔥 Auto exit if goodbye intent detected
        if ints[0]['intent'] == "goodbye":
            print("Bot session ended.")
            break


chat()
        
