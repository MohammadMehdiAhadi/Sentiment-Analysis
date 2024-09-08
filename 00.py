import pandas as pd
import chardet
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Detect encoding
with open("all_data.csv", "rb") as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    print(f"Detected encoding: {encoding}")

# Load CSV using detected encoding
column_names = ['Sentiment', 'Text']
df = pd.read_csv("all_data.csv", names=column_names, encoding=encoding)

# Split data into features (X) and labels (y)
X = df["Text"]
y = df["Sentiment"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)

# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

# Initialize and train the MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(500, 300, 100,33), early_stopping=False)
model.fit(x_train_tfidf, y_train)

# Make predictions and evaluate the model
pred = model.predict(x_test_tfidf)
print(classification_report(y_test, pred))
# Define the mapping of class indices to sentiment labels
class_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Predict the probability distribution
text = ["Nordea Group's operating profit increased in 2010 by 18 percent year-on-year to 3.64 billion euros and total revenue by 3 percent to 9.33 billion euros."]
new_tfidf = vectorizer.transform(text)

# Get the predicted probabilities
probabilities = model.predict_proba(new_tfidf)

# Get the predicted class (the index of the max probability)
predicted_class = probabilities.argmax()

# Map the predicted class to the corresponding sentiment
predicted_sentiment = class_mapping[predicted_class]

print(f"The predicted sentiment is: {predicted_sentiment}")


