from sklearn.svm import SVC


def svm_maker():
    model = SVC(C=2.0, kernel='rbf')
    return model


def svm_trainer(x_train, y_train):
    model = svm_maker()
    new_model = model.fit(x_train, y_train)
    return new_model


def svm_prediction(model , x_test):
    new_model = model
    pred = new_model.predict(x_test)
    return pred
