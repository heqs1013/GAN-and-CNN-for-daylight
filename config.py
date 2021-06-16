# some training parameters
EPOCHS = 200
BATCH_SIZE = 32
NUM_CLASSES = 2
image_height = 256
image_width = 192
channels = 1
save_model_dir = "saved_model_static/model"
dataset_dir = "dataset_parametric/"
train_dir = dataset_dir + "train"
valid_dir = dataset_dir + "valid"
test_dir = dataset_dir + "test/"

# choose a network
# model = "resnet18"
# model = "resnet34"
model = "resnet50"
# model = "resnet101"
# model = "resnet152"
