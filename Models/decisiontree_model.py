from sklearn.tree import DecisionTreeClassifier


def decision_tree_maker():
    model = DecisionTreeClassifier(criterion='gini', min_samples_leaf= 3, min_samples_split= 4, splitter= 'random')
    return model


def decision_tree_trainer(x_train, y_train):
    model = decision_tree_maker()
    new_model = model.fit(x_train, y_train)
    return new_model


def decision_tree_prediction(x_train, y_train, x_test):
    model = decision_tree_trainer(x_train, y_train)
    pred = model.predict(x_test)
    return pred
