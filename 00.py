try:
    import numpy as np
    import pandas as pd
    import chardet
    from matplotlib import pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
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

    # Detect encoding
    with open("all_data.csv", "rb") as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        print(f"Detected encoding: {encoding}")

    # Load CSV using detected encoding
    column_names = ['Sentiment', 'Text']
    df = pd.read_csv("all_data.csv", names=column_names, encoding=encoding)

    print("Analizing data . . .")

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
    print("Done")
    print("Fitting data, please wait . . . ")
    logistic_model = logistic_trainer(x_train_tfidf, y_train)
    mlp_model = mlp_trainer(x_train_tfidf, y_train)
    knn_model = knn_trainer(x_train_tfidf, y_train)
    svm_model = svm_trainer(x_train_tfidf, y_train)
    decition_model = decision_tree_trainer(x_train_tfidf, y_train)
    random_forest_model = random_forest_trainer(x_train_tfidf, y_train)
    print("Done")
    print("Predicting data, please wait . . . ")

    predictions_stacking = np.vstack([logistic_prediction(logistic_model, x_test_tfidf),
                                      mlp_prediction(mlp_model, x_test_tfidf),
                                      knn_prediction(knn_model, x_test_tfidf),
                                      svm_prediction(svm_model, x_test_tfidf),
                                      decision_tree_prediction(decition_model, x_test_tfidf),
                                      random_forest_prediction(random_forest_model, x_test_tfidf)
                                      ]).T

    print("Done")
    print()
    print("Summary : ")
    print("________________________________________________________________")
    # Meta model prediction
    predictions_final = final_pred(predictions_stacking, y_test, predictions_stacking)
    accuracy = np.mean(predictions_final == y_test)
    print("دقت مدل Stacking:", accuracy * 100)
    print("________________________________________________________________")
    # Classification reports
    print("Logistic :")
    print(classification_report(y_test, logistic_prediction(logistic_model, x_test_tfidf), zero_division=1))
    print("________________________________________________________________")
    print("MLPClassifier :")
    print(classification_report(y_test, mlp_prediction(mlp_model, x_test_tfidf), zero_division=1))
    print("________________________________________________________________")
    print("KNN :")
    print(classification_report(y_test, knn_prediction(knn_model, x_test_tfidf), zero_division=1))
    print("________________________________________________________________")
    print("SVM :")
    print(classification_report(y_test, svm_prediction(svm_model, x_test_tfidf), zero_division=1))
    print("________________________________________________________________")
    print("DecisionTree :")
    print(classification_report(y_test, decision_tree_prediction(decition_model, x_test_tfidf), zero_division=1))
    print("________________________________________________________________")
    print("RandomForest :")
    print(classification_report(y_test, random_forest_prediction(random_forest_model, x_test_tfidf), zero_division=1))
    print("________________________________________________________________")
    print("Final :")
    print(classification_report(y_test, predictions_final))
    print("________________________________________________________________")

    # Create a heatmap for the confusion matrix

    print("Done")
    print()
    print("Visualizing . . .  ")
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
    print("Done")
    print()

    print("For Example :")
    print(
        "Biohit said that it will reduce the number of leased personnel by 10 , and lay off 10 of its own personnel .")
    print()
    models = [decition_model,
              svm_model,
              mlp_model,
              random_forest_model,
              knn_model,
              logistic_model
              ]
    for model in models:
        # Define the mapping of class indices to sentiment labels
        class_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
        # Predict the probability distribution
        text = [
            "Biohit said that it will reduce the number of leased personnel by 10 , and lay off 10 of its own personnel ."]
        new_tfidf = vectorizer.transform(text)

        # Get the predicted probabilities
        model = model
        probabilities = model.predict(new_tfidf)

        # Get the predicted class (the index of the max probability)
        predicted_class = probabilities.argmax()

        # Map the predicted class to the corresponding sentiment
        predicted_sentiment = class_mapping[predicted_class]

        print(f"The predicted sentiment of {model} is: {predicted_sentiment}")
        print()
        print("________________________________________________________________")
        print()

    while True:
        new_text = input("Type your text for predict : ")
        if new_text:
            for model in models:
                # Define the mapping of class indices to sentiment labels
                class_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
                # Predict the probability distribution
                text = [new_text]
                new_tfidf = vectorizer.transform(text)

                # Get the predicted probabilities
                model = model
                probabilities = model.predict(new_tfidf)

                # Get the predicted class (the index of the max probability)
                predicted_class = probabilities.argmax()

                # Map the predicted class to the corresponding sentiment
                predicted_sentiment = class_mapping[predicted_class]
                print("________________________________________________________________")
                print(f"The predicted sentiment of {model} is: {predicted_sentiment}")
                print()
                print("________________________________________________________________")
                print()

        else:
            print(" no text added ")
except Exception as e:
    print("Something went wrong !")
    print("Please try again ")
    print(e)
