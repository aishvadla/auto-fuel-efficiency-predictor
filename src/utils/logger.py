import logging
from pathlib import Path
from datetime import datetime

# Generate a unique log file name based on current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = Path.cwd() / "logs" / LOG_FILE

logs_path.mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH = logs_path / LOG_FILE

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
