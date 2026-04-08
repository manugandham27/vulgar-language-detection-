import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import sys
import warnings

# Add parent directory to path to import preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.text_preprocessor import TextPreprocessor

warnings.filterwarnings('ignore')

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found. Please run generate_dataset.py first.")
        sys.exit(1)
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['text', 'label'])
    return df['text'].values, df['label'].values

def train_models():
    X_raw, y_raw = load_data('data/dataset.csv')
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    # Save label encoder mapping
    label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    joblib.dump(label_encoder, 'outputs/models/label_encoder.joblib')
    print(f"Classes: {label_mapping}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define models to train
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, n_jobs=-1),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
    }
    
    best_model = None
    best_acc = 0
    best_model_name = ""
    
    results = {}

    preprocessor = TextPreprocessor()
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('vectorizer', vectorizer),
            ('classifier', model)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        
        # Save model
        model_path = f'outputs/models/{name.lower().replace(" ", "_")}.joblib'
        joblib.dump(pipeline, model_path)
        print(f"Model saved to {model_path}")
        
        # Track best model
        if acc > best_acc:
            best_acc = acc
            best_model = pipeline
            best_model_name = name
            
    print("\n--- Summary ---")
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")
        
    print(f"\nBest Model: {best_model_name} with accuracy {best_acc:.4f}")
    
    # Save the best model as 'best_model.joblib' for easy access
    joblib.dump(best_model, 'outputs/models/best_model.joblib')
    print("Best model saved to outputs/models/best_model.joblib")

if __name__ == "__main__":
    os.makedirs('outputs/models', exist_ok=True)
    train_models()
