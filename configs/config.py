import os

# Paths
RESOURCES_PATH = os.path.join("resources")
DATASET_ROOT_PATH = os.path.join(RESOURCES_PATH, "pixel_dataset")
DATASET_PATH = os.path.join(DATASET_ROOT_PATH, "images")
SPRITES_NPY_PATH = os.path.join(DATASET_ROOT_PATH, "sprites.npy")
SPRITES_LABELS_NPY_PATH = os.path.join(DATASET_ROOT_PATH, "sprites_labels.npy")

# Hyperparameters
# train_batch_size = 128
# num_epochs = 300
# latent_dim = 128
# learning_rate_G = 0.00015
# learning_rate_D = 0.00005
latent_dim = 256
img_size = 16
train_batch_size = 128
num_epochs = 800
learning_rate_G = 0.00005
learning_rate_D = 0.00001
