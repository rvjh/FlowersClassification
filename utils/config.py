import os

DATA_PATH = "data"
DATA_DIR = os.path.join(DATA_PATH, "flowers")
DATA_FOLDER = os.path.join(DATA_PATH,"split_folder")
TRAIN_DIR = os.path.join(DATA_FOLDER,"train")
TEST_DIR = os.path.join(DATA_FOLDER,"test")
VAL_DIR = os.path.join(DATA_FOLDER,"val")
IMAGE_SIZE = (240, 320, 3)
CLASSES = 5
BATCH_SIZE = 16
EPOCHS = 3
MODELDIR = r"D:\Practice\FlowersClassification\modelDir"
TRAINED_MODEL_DIR = os.path.join(MODELDIR, "models")
CHECKPOINT_DIR = os.path.join(MODELDIR, "checkpoints")
BASE_LOG_DIR = "baseLogDir"
TENSORBOARD_ROOT_LOG_DIR = os.path.join(BASE_LOG_DIR, "tensorboard_log_dir")
AUGMENTATION = True
