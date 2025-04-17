import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load dataset (replace 'climate_fever_dataset.csv' with actual dataset if needed)
df = pd.read_csv('cyberbullying_dataset.csv')

# Ensure proper columns exist
df = df[['claim', 'label']]
df = df.dropna()

# Convert labels to categorical if needed
if df['label'].dtype == 'object':
    df['label'] = df['label'].astype('category').cat.codes  # Convert categorical to numerical

# Text preprocessing function
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['clean_text'] = df['claim'].apply(clean_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM Model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Save the trained model and vectorizer
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))
pickle.dump(svm_model, open("svm_model.pkl", "wb"))

print("Model training complete! 'tfidf_vectorizer.pkl' and 'svm_model.pkl' are saved.")