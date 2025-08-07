from sklearn.linear_model import LogisticRegression

def logreg_model():
    best_params = {
        'C': 0.0038,            # Inverse of regularization strength
        'penalty': 'l2',                       # 'l1' can be used if solver='liblinear'
        'solver': 'liblinear',                 # solver must support the penalty type
        'class_weight': 'balanced',      
        'max_iter' : 1000         
    }

    model = LogisticRegression(**best_params)
    return model


'''param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],            # Inverse of regularization strength
    'penalty': ['l2'],                       # 'l1' can be used if solver='liblinear'
    'solver': ['liblinear'],                     # solver must support the penalty type
    'class_weight': ['balanced']             # To handle imbalance
}

grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='recall',n_jobs=-1)
grid.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid.best_params_)

# Best estimator
best_model = grid.best_estimator_'''