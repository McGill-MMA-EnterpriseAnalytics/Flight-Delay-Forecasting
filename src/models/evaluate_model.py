from sklearn.metrics import f1_score


def evaluate_model(model, X_val, y_val):
    preds = model.predict(X_val)
    return f1_score(y_val, preds)
