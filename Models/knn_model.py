from sklearn.neighbors import KNeighborsClassifier


def knn_maker():
    model = KNeighborsClassifier(algorithm='kd_tree', n_neighbors=20, weights='distance')
    return model


def knn_trainer(x_train, y_train):
    model = knn_maker()
    new_model = model.fit(x_train, y_train)
    return new_model


def knn_prediction(x_train, y_train, x_test):
    model = knn_trainer(x_train, y_train)
    pred = model.predict(x_test)
    return pred
