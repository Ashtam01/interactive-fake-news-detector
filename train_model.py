import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# --- 1. Define File Paths ---

DATA_FOLDER = 'data'
MODELS_FOLDER = 'models'
FAKE_CSV = os.path.join(DATA_FOLDER, 'Fake.csv')
TRUE_CSV = os.path.join(DATA_FOLDER, 'True.csv')
MODEL_PATH = os.path.join(MODELS_FOLDER, 'model.joblib')
VECTORIZER_PATH = os.path.join(MODELS_FOLDER, 'vectorizer.joblib')

# --- 2. Load and Preprocess Data ---
print("Starting script...")


if not (os.path.exists(FAKE_CSV) and os.path.exists(TRUE_CSV)):
    print(f"Error: Make sure 'Fake.csv' and 'True.csv' are in the '{DATA_FOLDER}' folder.")
    print("You can download them from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
    exit()

print("Loading data...")
try:
    df_fake = pd.read_csv(FAKE_CSV)
    df_true = pd.read_csv(TRUE_CSV)
except Exception as e:
    print(f"Error reading CSV files: {e}")
    exit()

# Add 'label' column: 0 for Fake, 1 for True
df_fake["label"] = 0
df_true["label"] = 1


print("Combining and preprocessing data...")

df_combined = pd.concat([df_fake, df_true], ignore_index=True)


df_combined = df_combined.drop_duplicates(subset=['text'])



df_combined = df_combined.drop(['title', 'subject', 'date'], axis=1, errors='ignore')



df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)


df_combined.dropna(subset=['text'], inplace=True)

print(f"Data loaded. Total articles for training: {len(df_combined)}")

# --- 3. Text Cleaning Function ---
def clean_text(text):
    """
    1. Converts to lowercase
    2. Removes punctuation
    3. Strips extra whitespace
    """
    if not isinstance(text, str):

        return ""
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) 
    text = " ".join(text.split()) 
    return text

print("Cleaning text data...")
df_combined['text'] = df_combined['text'].apply(clean_text)
print("Text cleaning complete.")

# --- 4. Define Features (X) and Target (y) ---

X = df_combined['text']
y = df_combined['label']

# --- 5. Train-Test Split ---
print("Splitting data into train and test sets (75% train, 25% test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# --- 6. Vectorization (TF-IDF) ---
print("Vectorizing text (TF-IDF)...")

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform on the training data
X_train_vec = vectorizer.fit_transform(X_train)

# Only transform the test data (using the vocab from the training data)
X_test_vec = vectorizer.transform(X_test)
print("Vectorization complete.")

# --- 7. Train the Model (Logistic Regression) ---
print("Training the Logistic Regression model...")

model = LogisticRegression(max_iter=1000, n_jobs=-1)
model.fit(X_train_vec, y_train)
print("Model training complete.")

# --- 8. Evaluate the Model ---
print("Evaluating model performance on the test set...")
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

# --- 9. Save the Model and Vectorizer ---


os.makedirs(MODELS_FOLDER, exist_ok=True) 

# Save the trained model
joblib.dump(model, MODEL_PATH)
# Save the vectorizer
joblib.dump(vectorizer, VECTORIZER_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Vectorizer saved to: {VECTORIZER_PATH}")
print("\n--- Training process finished successfully! ---")
