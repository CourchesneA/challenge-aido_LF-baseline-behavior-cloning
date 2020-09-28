import tensorflow as tf
from frankModel import FrankNet  # The model we are gonna use to train
from logReader import Reader
# from sklearn.model_selection import train_test_split
from log_schema import Episode, Step
import sys
import time
import numpy as np
import os
import datetime


#! Training Configuration
# EPOCHS = 1000000 #EPOCHS
EPOCHS = 25
INIT_LR = 1e-3   #LEARNING RATE
BS = 8          #Batch Size 
GPU_COUNT = 1    # Change this value if you are using multiple GPUs
MULTI_GPU = False #Change this to enable multi-GPU

#! Log Interpretation
STORAGE_LOCATION = "trained_models/"
#DATA_FILE = "training_data.log"
DATA_FILE = "train_750_1000.log"

# #! Global training data storage
# # TODO: This should be optimized?
# observation = []
# linear = []
# angular = []


# def load_data():
#     global observation, linear, angular
#     reader = Reader('train_logA.log')
#     observation, linear, angular = reader.read()
#     observation = np.array(observation)
#     linear = np.array(linear)
#     angular = np.array(angular)
#     print('Observation Length: ', len(observation))
#     print('Linear Length: ', len(linear))
#     print('Angular Length: ', len(angular))
#     # exit()
#     return

def load_dataset() -> (tf.data.Dataset, int):
    """
    returns dataset, datasetsize
    """
    reader = Reader(DATA_FILE)
    dataset = tf.data.Dataset.from_generator(
        reader.get_dataset,
        (tf.float32, tf.float32)
    )
    return (dataset, reader.observations_size)




# -----------------------------------------------------------------------------
# Define custom loss functions for regression in Keras
# -----------------------------------------------------------------------------
# root mean squared error (rmse) for regression
def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# mean squared error (mse) for regression
def mse(y_true, y_pred):
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)

# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


#!================================================================
# 0. Create directories
try:
    os.makedirs("trainedModel")
except FileExistsError:
    print("Directory already exists!")
except OSError:
    print("Create folder for trained model failed. Please check system permissions.")
    exit()    

# 1. Load all the datas
dataset, ds_size = load_dataset()
dataset = dataset.batch(1,drop_remainder=True).prefetch(1)
print('Load generator complete')

# for (x,y) in dataset:
#     print("Data shape: ", x.shape, y.shape)
#     break
# sys.exit(0)
# Data shape:  (480, 640, 3) (2,)


# 2. Split training and testing
# observation_train, observation_valid, linear_train, linear_valid, angular_train, angular_valid = train_test_split(
#     observation, linear, angular, test_size=0.2, shuffle=True)

train_size = int(0.8 * ds_size)

train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

# 3. Build the model
# single_model = FrankNet.build(200, 150)
single_model = FrankNet.build(480, 640)

# 4. Define the loss function and weight
losses = {
    "Linear": "mse",
    "Angular": "mse"
}
lossWeights = {"Linear": 0.001, "Angular": 10.0}

# 5. Select optimizer
opt = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# 6. Select loss metrics
metrics_list = ["mse", rmse, r_square]

# 7. Select if multi GPU training or single GPU training.
if MULTI_GPU:
    print("Currently using multiple GPUs")
    model = tf.keras.utils.multi_gpu_model(single_model, gpus=GPU_COUNT) #TODO: Fix using the tf.distribute
else:
    print("Currently using single GPUs")
    model = single_model

# 8. Compile model and plot to see
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
              metrics=metrics_list)

# 9. Setup tensorboard
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='logs/{}'.format(datetime.now().strftime("%Y%m%d-%H%M%S")))

# 10. checkpoint
#? Keep track of the best validation loss model
filepath1 = "trainedModel/FrankNetBest2_Validation.h5"
checkpoint1 = tf.keras.callbacks.ModelCheckpoint(
    filepath1, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

#? Keep track of the best loss model
filepath2 = "trainedModel/FrankNetBest2_Loss.h5"
checkpoint2 = tf.keras.callbacks.ModelCheckpoint(
    filepath2, monitor='loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint1, checkpoint2, tensorboard]

# 11. GO!
# history = model.fit(observation_train,
#                     {"Linear": linear_train,
#                         "Angular": angular_train}, validation_data=(observation_valid, {
#                             "Linear": linear_valid, "Angular": angular_valid}),
#                     epochs=EPOCHS, callbacks=callbacks_list, verbose=1)

print("Training starting...")
history = model.fit(train_dataset, validation_data=(test_dataset),
                    epochs=EPOCHS, callbacks=callbacks_list, verbose=2)
print("Done, Saving model...")

model.save('trainedModel/TNet.h5')
print("Saved model TNet.h5")