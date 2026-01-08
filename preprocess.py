import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re

print("=== Phase 2: Data Preprocessing ===")

# 1. LOAD DATA
print("1. Loading dataset...")
fake = pd.read_csv('data/Fake.csv')
true = pd.read_csv('data/True.csv')

# 2. CREATE LABELS
fake['label'] = 1  # Fake
true['label'] = 0  # Real
df = pd.concat([fake, true], ignore_index=True)

print(f"Dataset: {len(df):,} articles")

# 3. CLEAN + COMBINE
def clean_text(text):
    if pd.isna(text): return ""
    text = re.sub(r'\s+', ' ', str(text))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip().lower()

df['content'] = (df['title'] + ' ' + df['text']).apply(clean_text)
df = df[['content', 'label']].dropna()

print(f"Cleaned: {len(df):,} articles")

# 4. SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    df['content'], df['label'], 
    test_size=0.2, random_state=42, stratify=df['label']
)

# 5. TF-IDF
print("TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Features: {X_train_tfidf.shape}")

# 6. SAVE
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(X_train_tfidf, 'X_train.joblib')
joblib.dump(X_test_tfidf, 'X_test.joblib')
joblib.dump(y_train, 'y_train.joblib')
joblib.dump(y_test, 'y_test.joblib')

print("✅ Phase 2 COMPLETE - Ready for training!")
