import os
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from joblib import dump

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import(
    accuracy_score,
    classification_report,
    recall_score,
    f1_score
)

def train_model():
    try:
        load_dotenv()

        PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()

        DATASET_PATH = PROJECT_ROOT / os.getenv("DATASET_DIR") / os.getenv("DATASET_NAME")
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
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(LOG_PATH)
            ]
        )

        # Loading dataset
        df = pd.read_csv(DATASET_PATH)
        logging.info(f"Dataset loaded with shape: {df.shape}")

        # separate X and y
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        # create a signature for each feature row - to prevent duplicates flow to test set
        row_signature = pd.util.hash_pandas_object(X, index=False)

        # Split data into train and test using GroupShuffleSplit
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        train_idx, test_idx = next(gss.split(X, y, groups=row_signature))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        print("Train Shape:", X_train.shape)
        print("Test Shape:", X_test.shape)

        logging.info(f"Training Shape: {X_train.shape}, Test Shape: {X_test.shape}")

        best_rf = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            bootstrap=True,
            ccp_alpha=0.0017,
            max_depth=5,
            max_features="sqrt",
            max_samples=0.6,
            min_samples_leaf=11,
            min_samples_split=30,
            n_estimators=1119
        )

        # keep scaler in pipeline to match nootebook structure
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", best_rf)
            ]
        )

        pipeline.fit(X_train, y_train)
        logging.info("Model training completed")


        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        # accuracy
        train_acc = accuracy_score(y_train, y_train_pred) * 100
        test_acc = accuracy_score(y_test, y_test_pred) * 100

        # recall
        train_recall = recall_score(y_train,y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)

        # f1 score
        train_f1 = f1_score(y_train,y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        # results
        logging.info(f"Training Accuracy:> {train_acc:.3f} | Recall:> {train_recall:.3f} | F1 Score:> {train_f1:.3f}")
        logging.info(f"Test Accuracy:> {test_acc:.3f} | Recall:> {test_recall:.3f} | F1 Score:> {test_f1:.3f}")

        logging.info("Training Classification Report:\n" + classification_report(y_train,y_train_pred))
        logging.info("Test Classification Report:\n" + classification_report(y_test,y_test_pred))

        # save trained model
        dump(pipeline, MODEL_PATH)
        logging.info(f"Model save to {MODEL_PATH}")
        logging.info("Training completed....")

    except Exception as e:
        print(f"Training Failed: {e}")
        logging.exception(f"Training Script Failed: {e}")
        raise    

if __name__ == "__main__":
    train_model()