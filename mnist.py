import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

############################################################################
# SETUP
############################################################################

#Load the data
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

#Scale the images
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

#Add black/white channel dimension
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

#One-hot encode targets
num_classes = np.unique(Y_train).shape[0]
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)

#Define the model, loss, and optimizer
def model_constructor():
    model = keras.Sequential([
            keras.Input(shape=X_train.shape[1:]),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax")
        ])
    
    sgd_optimizer = keras.optimizers.SGD(learning_rate=0.1)
    model.compile(loss="categorical_crossentropy", optimizer=sgd_optimizer, metrics=["accuracy"])
    
    return model

############################################################################
# VANILLA SGD
############################################################################

#Train the model for 10 epochs
model = model_constructor()
model.summary()
model.fit(X_train, Y_train, batch_size=64, epochs=10)

#Evaluate the model on the test set
score = model.evaluate(X_test, Y_test)
print("Vanilla SGD - Test loss: {0:.4f}; Test accuracy: {1:.4f}".format(*score))

############################################################################
# FEDERATED AVERAGING (McMahan, Brendan, et al. 2017)
############################################################################

#------------------------------------------------------
#The following variables are named to match the paper.
#Note: in this paper workers are called "clients"

K = 10  # K is the number of workers
C = 0.3 # C is the fraction of workers randomly selected per round
E = 1   # E is the number of epochs a worker will run over its data per round
B = 64  # B is the worker minibatch size
#------------------------------------------------------

T = 5   # Number of rounds of FederatedAveraging to run

#Partition the dataset into K splits

#Note: In the real world we would not have access to the dataset as it would be distributed
#      across all the worker devices. Here in simulation, we have access to the complete dataset
#      and define the splits that go to each worker.

X_train_splits = np.array_split(X_train, K)
Y_train_splits = np.array_split(Y_train, K)

#Create K worker models
workers = [model_constructor() for i in range(K)]

#Create the global model
global_model = model_constructor()
global_weights = global_model.get_weights()

#Execute the rounds of FederatedAveraging

#Note: In the real world each worker would perform its update in parallel on a separate device.
#      Here in simulation, we can perform worker updates sequentially on the same device.
rng = np.random.default_rng(seed=100)
m = int(np.ceil(C*K)) # Number of clients per round

for t in range(T):
    #Randomly pick the workers to be used for this round
    worker_indices = rng.integers(K, size=m)
    
    #Perform the local update on each worker starting from the global weights
    worker_weights = []
    for wk_i in worker_indices:
        worker = workers[wk_i]
        worker.set_weights(global_weights)
        worker.fit(X_train_splits[wk_i], Y_train_splits[wk_i], batch_size=B, epochs=E)
        #Return the worker weights to the "server"
        worker_weights.append(worker.get_weights())
        
    #Average all the worker weights to get the updated global weights
    for i in range(len(global_weights)):
        global_weights[i] = np.mean([w[i] for w in worker_weights], axis=0)
    
global_model.set_weights(global_weights)

#Evaluate the global model on the test set
score = global_model.evaluate(X_test, Y_test)
print("FederatedAveraging - Test loss: {0:.4f}; Test accuracy: {1:.4f}".format(*score))