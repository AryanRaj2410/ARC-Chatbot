"""
ARC Chatbot — Training Script
Run this locally:  python train.py
Requirements:      pip install tensorflow nltk numpy
"""

import random
import json
import pickle

import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping

# ── NLTK setup ─────────────────────────────────────────────────────────────────
for pkg in ['punkt', 'punkt_tab', 'wordnet', 'omw-1.4']:
    nltk.download(pkg, quiet=True)

lemmatizer  = WordNetLemmatizer()
IGNORE      = set(['?', '!', '.', ',', ';', ':', "'s"])

# ── Load intents ───────────────────────────────────────────────────────────────
with open('intents.json') as f:
    intents = json.load(f)

words  = []
labels = []
docs   = []

for intent in intents['intents']:
    tag = intent['tag']
    if tag not in labels:
        labels.append(tag)
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        docs.append((tokens, tag))

words  = sorted(set(
    lemmatizer.lemmatize(w.lower())
    for w in words
    if w not in IGNORE
))
labels = sorted(labels)

print(f"✅ Vocabulary size : {len(words)} words")
print(f"✅ Intent classes  : {len(labels)} tags")
print(f"✅ Training samples: {len(docs)} patterns\n")

pickle.dump(words,  open('words.pkl',  'wb'))
pickle.dump(labels, open('labels.pkl', 'wb'))

# ── Build bag-of-words training data ──────────────────────────────────────────
training = []
zero_out = [0] * len(labels)

for pattern_words, tag in docs:
    lemmatized = [lemmatizer.lemmatize(w.lower()) for w in pattern_words]
    bag        = [1 if w in lemmatized else 0 for w in words]
    output     = list(zero_out)
    output[labels.index(tag)] = 1
    training.append([bag, output])

random.shuffle(training)
training = np.array(training, dtype=object)

x_train = np.array(list(training[:, 0]), dtype=np.float32)
y_train = np.array(list(training[:, 1]), dtype=np.float32)

# ── Build model ────────────────────────────────────────────────────────────────
model = Sequential([
    Dense(256, input_shape=(x_train.shape[1],), activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64,  activation='relu'),
    Dropout(0.3),
    Dense(y_train.shape[1], activation='softmax'),
])

model.summary()

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy']
)

# ── Early stopping to avoid overfitting ───────────────────────────────────────
early_stop = EarlyStopping(
    monitor='accuracy',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

# ── Train ──────────────────────────────────────────────────────────────────────
print("\n🚀 Starting training...\n")
history = model.fit(
    x_train, y_train,
    epochs=500,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1
)

# ── Save ───────────────────────────────────────────────────────────────────────
model.save('chatbot_Application_model.h5')
print("\n✅ Model saved  →  chatbot_Application_model.h5")
print("✅ Vocab saved  →  words.pkl")
print("✅ Labels saved →  labels.pkl")
print("\nAll done! You can now run your chatbot app.")
