import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.text_preprocessor import TextPreprocessor

def load_data():
    df = pd.read_csv('data/dataset.csv')
    df = df.dropna(subset=['text', 'label'])
    return df['text'].values, df['label'].values

def evaluate_best_model():
    print("Loading data...")
    X_raw, y_raw = load_data()
    
    print("Loading model and label encoder...")
    model_path = 'outputs/models/best_model.joblib'
    encoder_path = 'outputs/models/label_encoder.joblib'
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        print("Model or encoder not found. Please run train.py first.")
        sys.exit(1)
        
    pipeline = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    
    y_true = label_encoder.transform(y_raw)
    
    print("Making predictions on the full dataset for evaluation visualization...")
    y_pred = pipeline.predict(X_raw)
    
    class_names = label_encoder.classes_
    
    # 1. Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:\n")
    print(report)
    
    with open('outputs/reports/classification_report.txt', 'w') as f:
        f.write(report)
        
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Best Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('outputs/figures/confusion_matrix.png', dpi=300)
    plt.close()
    
    print("\nEvaluation complete. Check 'outputs/reports/' and 'outputs/figures/'.")

if __name__ == "__main__":
    os.makedirs('outputs/reports', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    evaluate_best_model()
