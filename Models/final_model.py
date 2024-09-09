from sklearn.neighbors import KNeighborsClassifier as Knn



def final_model_maker():
    model = Knn(algorithm='auto', n_neighbors=8, weights='distance')
    return model


def final_fit(x_train, y_train):
    model = final_model_maker()
    model.fit(x_train, y_train)
    return model


def final_pred(x_train, y_train, x_test):
    model = final_fit(x_train, y_train)
    pred = model.predict(x_test)
    return pred

