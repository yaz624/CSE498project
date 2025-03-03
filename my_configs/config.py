import os

# Paths
RESOURCES_PATH = os.path.join("resources")
DATASET_ROOT_PATH = os.path.join(RESOURCES_PATH, "pixel_dataset")
DATASET_PATH = os.path.join(DATASET_ROOT_PATH, "images")
SPRITES_NPY_PATH = os.path.join(DATASET_ROOT_PATH, "sprites.npy")
SPRITES_LABELS_NPY_PATH = os.path.join(DATASET_ROOT_PATH, "sprites_labels.npy")

# Hyperparameters
train_batch_size = 32
num_epochs = 150
latent_dim = 100
