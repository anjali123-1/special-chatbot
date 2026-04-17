# special-chatbot
Below is a complete, working ML pipeline in Python for your project. It includes:

Data loading (CSV or sample dataset)

NLP preprocessing

Feature engineering (text + structured data)

Model training (Random Forest)

Evaluation

Saving model for deployment (FastAPI ready)


You can run this in Jupyter Notebook or VS Code.


---

🧠 1. Install Required Libraries

pip install pandas numpy scikit-learn nltk joblib


---

📊 2. Sample Dataset Format (CSV)

Create a file: mental_health_data.csv

text,stress_score,sleep_hours,social_interaction,risk_level
"I feel very anxious and tired",8,4,2,High
"I am doing okay but a bit stressed",5,6,5,Medium
"I feel happy and relaxed",2,8,7,Low
"I feel lonely and depressed",9,3,1,High
"Just normal day, nothing special",3,7,6,Low


---

🧾 3. Complete ML Code

import pandas as pd
import numpy as np
import nltk
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

nltk.download('stopwords')
from nltk.corpus import stopwords
import re

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("mental_health_data.csv")

# -----------------------------
# 2. Text Preprocessing
# -----------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)

# -----------------------------
# 3. Feature Engineering
# -----------------------------
# Text features (TF-IDF)
tfidf = TfidfVectorizer(max_features=100)
X_text = tfidf.fit_transform(df['clean_text']).toarray()

# Structured features
X_structured = df[['stress_score', 'sleep_hours', 'social_interaction']].values

# Combine features
X = np.hstack((X_text, X_structured))

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df['risk_level'])

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Model Training
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 6. Evaluation
# -----------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# -----------------------------
# 7. Save Model & Components
# -----------------------------
joblib.dump(model, "risk_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Model saved successfully!")


---

🔍 4. Prediction Code (Use After Training)

def predict_risk(text, stress, sleep, social):
    # Load saved files
    model = joblib.load("risk_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    le = joblib.load("label_encoder.pkl")

    # Clean text
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        words = [w for w in words if w not in stop_words]
        return " ".join(words)

    clean = clean_text(text)

    # Transform
    text_vec = tfidf.transform([clean]).toarray()
    structured = np.array([[stress, sleep, social]])

    final_input = np.hstack((text_vec, structured))

    # Predict
    pred = model.predict(final_input)
    return le.inverse_transform(pred)[0]


# Example
result = predict_risk(
    "I feel very stressed and anxious",
    stress=9,
    sleep=3,
    social=2
)

print("Predicted Risk Level:", result)
