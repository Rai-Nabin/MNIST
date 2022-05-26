import os
import torch
TRAIN_DATASET_PATH = os.path.join("data", "trainingSet")
MODEL_PATH = os.path.join("model", "model.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_CLASSES = 10
NUM_EPOCHS = 2
INIT_LR = 0.001
