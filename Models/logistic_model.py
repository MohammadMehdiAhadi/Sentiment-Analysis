from sklearn.linear_model import LogisticRegression


def logistic_maker():
    model = LogisticRegression(C=7, penalty='l1', solver='liblinear',max_iter=3000)
    return model


def logistic_trainer(x_train, y_train):
    model = logistic_maker()
    new_model = model.fit(x_train, y_train)
    return new_model


def logistic_prediction(model , x_test):
    new_model = model
    pred = new_model.predict(x_test)
    return pred
