import os.path

# TEMP, change
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'data')
RAW_DATA_PATH = os.path.join(DATA_PATH, 'zip')
MODEL_PATH = os.path.join(DATA_PATH, 'models')

WAV_NORM_FACTOR = 32768.0
SAMPLING_RATE = 16000

NUM_CLASSES = 30

OPT_EPOCHS = 25
OPT_LEARNING_RATE = 0.01
OPT_DECAY = OPT_LEARNING_RATE/OPT_EPOCHS

BATCH_SIZE = 32
EPOCHS = 10
STEPS_PER_EPOCH = 50
VALIDATION_STEPS = 10

