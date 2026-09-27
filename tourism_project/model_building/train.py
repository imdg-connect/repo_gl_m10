import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score, accuracy_score

def load_splits():
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze()
    ytest = pd.read_csv("ytest.csv").squeeze()
    return Xtrain, Xtest, ytrain, ytest

def build_pipeline(Xtrain):
    cat_cols = Xtrain.select_dtypes(include="object").columns.tolist()
    num_cols = Xtrain.select_dtypes(include="number").columns.tolist()

    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), cat_cols),
        (StandardScaler(), num_cols),
    )

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )

    pipe = make_pipeline(preprocessor, model)
    return pipe

def train():
    mlflow.set_experiment("tourism-wellness-package")

    Xtrain, Xtest, ytrain, ytest = load_splits()
    pipe = build_pipeline(Xtrain)

    param_grid = {
        "xgbclassifier__n_estimators": [100, 200],
        "xgbclassifier__max_depth": [3, 5, 7],
        "xgbclassifier__learning_rate": [0.05, 0.1],
    }

    with mlflow.start_run():
        grid = GridSearchCV(pipe, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1)
        grid.fit(Xtrain, ytrain)

        print("=== Best Parameters ===")
        print(grid.best_params_)
        mlflow.log_params(grid.best_params_)

        best_model = grid.best_estimator_
        ypred = best_model.predict(Xtest)

        acc = accuracy_score(ytest, ypred)
        f1 = f1_score(ytest, ypred)
        print("=== Evaluation ===")
        print(classification_report(ytest, ypred))

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(best_model, "model")

        # Save into deployment/ so the pipeline can commit it to the repo
        joblib.dump(best_model, "tourism_project/deployment/best_model.joblib")
        print("✅ Best model saved to tourism_project/deployment/best_model.joblib")

    return best_model

if __name__ == "__main__":
    train()
