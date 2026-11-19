import os
from pathlib import Path
BASE=Path(__file__).resolve().parent
API_KEY='AIzaSyAr4IYnykpwovqOJWzfBd7abVdAma_Ig3Q'
DATABASE_URL='https://diet-planner-3bdf3-default-rtdb.firebaseio.com'
ROOT='ks5785'
MODEL_DIR=Path(os.environ.get('MODEL_DIR',str(BASE/'models'))).resolve()
DATA_DIR=Path(os.environ.get('DATA_DIR',str(BASE/'data'))).resolve()
