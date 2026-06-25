from pathlib import Path

RANDOM_STATE=42
TEST_SIZE=0.2
MAX_ITER=1000
BASE_DIR=Path(__file__).resolve().parent.parent
DATA_PATH=BASE_DIR /"data" / "raw" /"diabetes.csv"
MODEL_PATH=BASE_DIR / "models"
#DEFAULT_MODEL =MODEL_PATH /'LogisticRegression.joblib'
DEFAULT_MODEL =MODEL_PATH /'RandomForestClassifier.joblib'