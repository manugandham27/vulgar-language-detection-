import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# We assume nltk packages have been downloaded
# nltk.download('stopwords')
# nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # We don't want to remove words that might be important for vulgarity/abuse detection
        vulgar_stopwords = {'not', 'no', 'are', 'is', 'you', 'he', 'she', 'they', 'we', 'I', 'me', 'my', 'your', 'yours'}
        self.stop_words = self.stop_words - vulgar_stopwords

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
            
        # 1. Lowercase
        text = text.lower()
        
        # 2. Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # 3. Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # 4. Remove special characters and numbers, keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # 5. Tokenize and remove extra whitespace
        tokens = text.split()
        
        # 6. Remove stopwords and lemmatize
        cleaned_tokens = []
        for word in tokens:
            if word not in self.stop_words:
                lemmatized_word = self.lemmatizer.lemmatize(word)
                cleaned_tokens.append(lemmatized_word)
                
        return ' '.join(cleaned_tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [self.clean_text(text) for text in X]
