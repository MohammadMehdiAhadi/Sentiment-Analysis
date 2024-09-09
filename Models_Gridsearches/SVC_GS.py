import chardet
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

with open("../all_data.csv", "rb") as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    print(f"Detected encoding: {encoding}")

# Load CSV using detected encoding
column_names = ['Sentiment', 'Text']
df = pd.read_csv("../all_data.csv", names=column_names, encoding=encoding)

# Split data into features (X) and labels (y)
X = df["Text"]
y = df["Sentiment"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)

vectorizer = TfidfVectorizer(max_features=5000)
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

params = {"C": [3.0, 2.0, 1.0, 0.5],
          "kernel": ['poly', 'rbf']
          }
model = GridSearchCV(estimator=SVC(tol=1e-5, gamma='scale'),
                     param_grid=params,
                     scoring="accuracy",
                     )
model.fit(x_train_tfidf, y_train)
model.predict(x_test_tfidf)
print(model.best_params_)
print(model.best_score_)

# {'C': 2.0, 'kernel': 'rbf'}
# 0.7674829492312705
