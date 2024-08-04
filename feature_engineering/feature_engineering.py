from sklearn.feature_selection import SelectKBest, f_classif

def select_features(X, y, k=10):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    return X_selected, selected_features