from sklearn.neural_network import MLPClassifier


def mlp_maker():
    model = MLPClassifier(activation='relu',solver='adam',early_stopping=False)
    return model


def mlp_trainer(x_train, y_train):
    model = mlp_maker()
    new_model = model.fit(x_train, y_train)
    return new_model


def mlp_prediction(model , x_test):
    new_model = model
    pred = new_model.predict(x_test)
    return pred
