# ARC Chatbot

A conversational AI chatbot built with Python, Flask, and TensorFlow using Natural Language Processing (NLP) techniques.

---

## 📁 Project Structure

```
ARC Chatbot/
│
├── templates/
│   └── index.html                  # Frontend chat interface (HTML/CSS/JS)
│
├── app.py                          # Flask web server & API routes
├── ChatBot.ipynb                   # Jupyter notebook for exploration & testing
├── ChatBot_Application.py          # Main chatbot application logic
├── chatbot_Application_model.h5    # Trained Keras/TensorFlow model
├── chatBot_model_file.py           # Model architecture & training script
├── intents.json                    # Intent definitions (tags, patterns, responses)
├── labels.pkl                      # Serialized intent label classes
└── words.pkl                       # Serialized vocabulary (bag-of-words)
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow flask nltk numpy
```

### Training the Model

Run this once to generate the model and pickle files:

```bash
python chatBot_model_file.py
```

This will produce:
- `chatbot_Application_model.h5` — the trained neural network
- `words.pkl` — the vocabulary
- `labels.pkl` — the intent labels

### Running the App

```bash
python app.py
```

Then open your browser and go to:

```
http://localhost:5000
```

---

## 🧠 How It Works

1. **intents.json** defines the chatbot's knowledge — each intent has patterns (user inputs) and responses.
2. **chatBot_model_file.py** tokenizes and lemmatizes patterns, builds a bag-of-words, and trains a neural network.
3. **ChatBot_Application.py** loads the trained model and predicts the best intent for any user message.
4. **app.py** serves the Flask web app and exposes a `/get` POST endpoint for the frontend to call.
5. **templates/index.html** provides the chat UI that communicates with the Flask backend.

---

## 🛠 Tech Stack

| Layer       | Technology                        |
|-------------|-----------------------------------|
| Frontend    | HTML, CSS, JavaScript             |
| Backend     | Python, Flask                     |
| ML Model    | TensorFlow / Keras (Dense + SGD)  |
| NLP         | NLTK (tokenization, lemmatization)|
| Data Format | JSON, Pickle                      |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
