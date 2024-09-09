import numpy as np
import pandas as pd
import chardet
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
from Models.mlp_model import *
from Models.knn_model import *
from Models.randomforest_model import *
from Models.logistic_model import *
from Models.svm_model import *
from Models.decisiontree_model import *
from Models.final_model import *
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.preprocessing import label_binarize


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

# Encode sentiment labels (e.g., 'positive', 'neutral', 'negative') to numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.1, random_state=0, stratify=y_encoded)


# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

predictions_stacking = np.vstack([logistic_prediction(x_train_tfidf, y_train, x_test_tfidf),
                                  mlp_prediction(x_train_tfidf, y_train, x_test_tfidf),
                                  knn_prediction(x_train_tfidf, y_train, x_test_tfidf),
                                  svm_prediction(x_train_tfidf, y_train, x_test_tfidf),
                                  decision_tree_prediction(x_train_tfidf, y_train, x_test_tfidf),
                                  random_forest_prediction(x_train_tfidf, y_train, x_test_tfidf)
                                  ]).T

# Meta model prediction
predictions_final = final_pred(predictions_stacking, y_test, predictions_stacking)
accuracy = np.mean(predictions_final == y_test)
print("دقت مدل Stacking:", accuracy * 100)
print("________________________________________________________________")
# Classification reports
print("MLPClassifier :")
print(classification_report(y_test, mlp_prediction(x_train_tfidf, y_train, x_test_tfidf), zero_division=1))
print("________________________________________________________________")
print("Logistic :")
print(classification_report(y_test, logistic_prediction(x_train_tfidf, y_train, x_test_tfidf), zero_division=1))
print("________________________________________________________________")
print("KNN :")
print(classification_report(y_test, knn_prediction(x_train_tfidf, y_train, x_test_tfidf), zero_division=1))
print("________________________________________________________________")
print("SVM :")
print(classification_report(y_test, svm_prediction(x_train_tfidf, y_train, x_test_tfidf), zero_division=1))
print("________________________________________________________________")
print("DecisionTree :")
print(classification_report(y_test, decision_tree_prediction(x_train_tfidf, y_train, x_test_tfidf), zero_division=1))
print("________________________________________________________________")
print("RandomForest :")
print(classification_report(y_test, random_forest_prediction(x_train_tfidf, y_train, x_test_tfidf), zero_division=1))
print("________________________________________________________________")
print("Final :")
print(classification_report(y_test, predictions_final))
print("________________________________________________________________")

# Create a heatmap for the confusion matrix

print("Done")
conf_matrix = confusion_matrix(y_test, predictions_final)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("final_predict.jpg")
plt.show()

# Bar Plot for Correct and Incorrect Predictions
correct_predictions = np.sum(predictions_final == y_test)
incorrect_predictions = np.sum(predictions_final != y_test)

plt.bar(['Correct', 'Incorrect'], [correct_predictions, incorrect_predictions], color=['green', 'red'])
plt.title("Correct vs Incorrect Predictions")
plt.xlabel("Prediction Type")
plt.ylabel("Number of Predictions")
plt.savefig("final_predict_bar_chart.jpg")
plt.show()

# # Binarize the output labels for multi-class
# y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])  # Assuming you have 3 classes
# n_classes = y_test_binarized.shape[1]
#
# # Calculate ROC curve and ROC AUC for each class
# fpr = {}
# tpr = {}
# roc_auc = {}
#
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], predictions_final[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#
# # Compute micro-average ROC curve and ROC AUC
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), predictions_final.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
# # Plot ROC curves for each class
# plt.figure()
# for i in range(n_classes):
#     plt.plot(fpr[i], tpr[i], label='Class {0} ROC curve (AUC = {1:0.2f})'.format(i, roc_auc[i]))
#
# plt.plot(fpr["micro"], tpr["micro"], label='Micro-average ROC curve (AUC = {0:0.2f})'.format(roc_auc["micro"]))
# plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic for Multiclass')
# plt.legend(loc="lower right")
# plt.savefig("multiclass_roc_auc.jpg")
# plt.show()

# # Define the mapping of class indices to sentiment labels
# class_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
# # Predict the probability distribution
# text = [
#     "Nordea Group's operating profit increased in 2010 by 18 percent year-on-year to 3.64 billion euros and total revenue by 3 percent to 9.33 billion euros."]
# new_tfidf = vectorizer.transform(text)
#
# # Get the predicted probabilities
# probabilities = model.predict_proba(new_tfidf)
#
# # Get the predicted class (the index of the max probability)
# predicted_class = probabilities.argmax()
#
# # Map the predicted class to the corresponding sentiment
# predicted_sentiment = class_mapping[predicted_class]
#
# print(f"The predicted sentiment is: {predicted_sentiment}")
