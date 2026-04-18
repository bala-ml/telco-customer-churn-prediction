import os
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    recall_score,
    f1_score,
)
from xgboost import XGBClassifier


def train_model():
    try:
        # load env file content to env vars
        load_dotenv()

        PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()

        DATASET_PATH = (
            PROJECT_ROOT / os.getenv("DATASET_DIR") / os.getenv("DATASET_NAME")
        )
        MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
        LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")

        TARGET_COL = os.getenv("TARGET_COL")
        TEST_SIZE = float(os.getenv("TEST_SIZE"))
        RANDOM_STATE = int(os.getenv("RANDOM_STATE"))

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH)],
        )

        # Load data
        df = pd.read_csv(DATASET_PATH)
        logging.info(f"Dataset loaded with shape: {df.shape}")

        # Separate X and y
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        y = y.map({"Yes": 1, "No": 0})
        X = X.drop(columns=["customerID"], errors="ignore")
        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )

        logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # Best params from notebook tuning
        best_xgb = XGBClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            learning_rate=0.1,
            max_depth=3,
            n_estimators=100,
            subsample=1.0,
            eval_metric="logloss",
        )

        # Keep scaler in pipeline to match notebook structure
        pipeline = Pipeline(steps=[("scaler", StandardScaler()), ("model", best_xgb)])

        pipeline.fit(X_train, y_train)
        logging.info("Model training completed")

        # Evaluation
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)

        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        logging.info(
            f"Train Accuracy: {train_acc:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}"
        )
        logging.info(
            f"Test  Accuracy: {test_acc:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}"
        )

        logging.info(
            "Train Classification Report:\n"
            + classification_report(y_train, y_train_pred)
        )
        logging.info(
            "Test Classification Report:\n" + classification_report(y_test, y_test_pred)
        )

        # save trained model
        dump(pipeline, MODEL_PATH)
        logging.info(f"Model saved to: {MODEL_PATH}")

        logging.info("Training Script Completed")

    except Exception as e:
        print(f"Training failed: {e}")
        logging.exception(f"Training Script Failed: {e}")
        raise


if __name__ == "__main__":
    train_model()
