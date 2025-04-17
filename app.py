from flask import Flask, request, jsonify, render_template
import pandas as pd
from fuzzywuzzy import process

app = Flask(__name__)

# Load dataset
file_path = "cyberbullying_dataset.csv"
df = pd.read_csv(file_path)

# Function to classify tweet
def classify_tweet(tweet_text):
    result = process.extractOne(tweet_text, df["tweet"].tolist())
    if result:
        match, score = result[0], result[1]
        if score > 80:  # Confidence threshold
            row = df[df["tweet"] == match].iloc[0]
            label = row["label"]

            if label == 1:
                sentiment = "Cyberbullying"
            elif label == -1:
                sentiment = "CyberStalking"
            else:
                sentiment = "Non-Cyberbullying"

            return sentiment
    return "Non-Cyberbullying"

@app.route("/")
def home():
    return render_template("index3.html")

@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    data = request.json
    tweet_text = data.get("tweet", "").strip()
    sentiment = classify_tweet(tweet_text)
    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=True)
