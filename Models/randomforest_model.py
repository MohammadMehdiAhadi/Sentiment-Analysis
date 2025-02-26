from sklearn.ensemble import RandomForestClassifier


def random_forest_maker():
    model = RandomForestClassifier(criterion='gini', min_samples_leaf=1, min_samples_split=2, n_estimators=300)
    return model


def random_forest_trainer(x_train, y_train):
    model = random_forest_maker()
    new_model = model.fit(x_train, y_train)
    return new_model


def random_forest_prediction(model , x_test):
    new_model = model
    pred = new_model.predict(x_test)
    return pred