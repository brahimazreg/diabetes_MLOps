import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("logs/predictions.jsonl")
LOG_FILE.parent.mkdir(exist_ok=True)

def log_prediction(input_data: dict, prediction: int, probability: float):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_data,
        "prediction": prediction,
        "probability": probability
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")