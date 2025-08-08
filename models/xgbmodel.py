from xgboost import XGBClassifier

def xgb_model(scale_pos_weight):
    best_params = {
        'colsample_bytree': 0.8,
        'learning_rate': 0.05,
        'max_depth': 3,
        'n_estimators': 40,
        'reg_alpha': 1,
        'reg_lambda': 1,
        'scale_pos_weight': float(scale_pos_weight),
        'subsample': 0.6
    }

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        **best_params
    )
    return model



# Train XGBoost and tune hyperparametes using gridsearch cv

'''param_grid = {
    'n_estimators': [40, 50],
    'max_depth': [2,3, 4],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.4, 0.6, 0.8],
    'reg_alpha': [0.1, 0.5, 1],
    'reg_lambda': [1, 5, 10],
    'scale_pos_weight': [1, scale_pos_weight]  # Based on your data
}

model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='f1',   # or 'roc_auc', 'accuracy', etc.
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Use best model
best_model = grid_search.best_estimator_


# Evaluate
y_pred = best_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

y_pred = best_model.predict(X_train)
print(confusion_matrix(y_train, y_pred))
print(classification_report(y_train, y_pred))'''
