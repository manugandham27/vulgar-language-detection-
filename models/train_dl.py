import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
import warnings

warnings.filterwarnings('ignore')

MAX_WORDS = 20000
MAX_LEN = 100
EMBEDDING_DIM = 100

def train_dl_model():
    print("=== Training Deep Learning Bi-LSTM Model ===")
    
    # 1. Load Data
    filepath = 'data/dataset.csv'
    if not os.path.exists(filepath):
        print("Dataset not found. Please run generate_dataset first.")
        sys.exit(1)
        
    df = pd.read_csv(filepath)
    df = df.dropna()
    X_raw = df['text'].values
    y_raw = df['label'].values
    
    # 2. Preprocess Labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    # We must format labels into categorical data
    num_classes = len(np.unique(y))
    y_categorical = tf.keras.utils.to_categorical(y, num_classes)
    
    # 3. Preprocess Text
    print("Tokenizing the text...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_raw)
    
    X_seq = tokenizer.texts_to_sequences(X_raw)
    X_padded = pad_sequences(X_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Save Tokenizer and Encoder immediately
    os.makedirs('outputs/models', exist_ok=True)
    joblib.dump(tokenizer, 'outputs/models/dl_tokenizer.joblib')
    joblib.dump(label_encoder, 'outputs/models/dl_label_encoder.joblib')
    
    # 4. Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
    )
    
    # 5. Build Bi-LSTM Architecture
    print("Building model architecture...")
    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    # 6. Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # 7. Training
    print("Starting Training (Epochs=10)...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    
    # 8. Evaluation & Save
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nFinal Test Accuracy: {accuracy*100:.2f}%")
    
    model.save('outputs/models/bilstm_model.h5')
    print("Deep Learning model securely saved to outputs/models/bilstm_model.h5")

if __name__ == '__main__':
    train_dl_model()
