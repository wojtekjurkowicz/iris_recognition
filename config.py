import os
import kagglehub


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
path = kagglehub.dataset_download("sondosaabed/casia-iris-thousand")
DATASET_PATH = os.path.join(path, "CASIA-Iris-Thousand", "CASIA-Iris-Thousand")
