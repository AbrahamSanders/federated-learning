import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

############################################################################
# SETUP
############################################################################
rng = np.random.default_rng(seed=100)

#Load the data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

#Scale the images
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

#Add black/white channel dimension
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

#Define the model, loss, and optimizer
def model_constructor():
    model = keras.Sequential([
            keras.Input(shape=X_train.shape[1:]),
            layers.Conv2D(32, kernel_size=(5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dense(10, activation="softmax")
        ])
    
    sgd_optimizer = keras.optimizers.SGD(learning_rate=0.1)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=sgd_optimizer, metrics=["accuracy"])
    
    return model

def split_training_data(X_train, y_train, num_splits, iid, rng):
    if iid:
        train_order = rng.permutation(X_train.shape[0])
    else:
        train_order = np.argsort(y_train)
        
    X_train = X_train[train_order]
    y_train = y_train[train_order]
    X_train_splits = np.array_split(X_train, num_splits)
    y_train_splits = np.array_split(y_train, num_splits)

    split_weights = [X_train_splits[i].shape[0] / X_train.shape[0] for i in range(num_splits)]  
    
    return X_train_splits, y_train_splits, split_weights

############################################################################
# VANILLA SGD
############################################################################

#Train the model for 10 epochs
model = model_constructor()
model.summary()
hist = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

############################################################################
# FEDERATED AVERAGING (McMahan, Brendan, et al. 2017)
############################################################################

def federated_averaging(X_train, y_train, X_val, y_val, K, C, E, B, MAX_T, iid, rng):
    """
    Trains a global model by running up to MAX_T rounds of FederatedAveraging over K workers.
    Returns the final global model and train/val metrics per communciation round.
    
    Parameters
    ----------
    X_train : numpy ndarray
        Training features of shape (n, d) for d-dimensional features
    y_train : numpy ndarray
        Training targets of shape (n, k) for k-dimensional targets
    X_val : numpy ndarray
        Validation features of shape (n, d) for d-dimensional features
    y_val : numpy ndarray
        Validation targets of shape (n, k) for k-dimensional targets
        
    The following variables are named to match the paper:
    
    K : int
        The number of workers
    C : float
        The fraction of workers randomly selected per round
    E : int
        The number of epochs a worker will run over its data per round
    B : int
        The worker minibatch size
    
    The following variables are not assigned explicit names in the paper:
    
    MAX_T : int
        Maximum number of rounds of FederatedAveraging to run
    iid : bool
        If iid, we simulate an evenly distributed random split of the data across
        workers. Otherwise each worker gets data in only one (or few) classes
    rng : numpy random._generator.Generator
        The random generator to use

    Returns
    -------
    global_model : Keras Model
        The final output model after FederatedAveraging is complete
    communication_rounds : dict
        Dictionary containing per-communiction-round metrics
    """
    
    #Partition the dataset into K splits
    
    #Note: In the real world we would not have access to the dataset as it would be distributed
    #      across all the worker devices. Here in simulation, we have access to the complete dataset
    #      and define the splits that go to each worker.
    
    #split_weights is the weight for each split (n_k / n) used for the weighted averaging
    #of worker weights at the end of each round of FederatedAveraging
    X_train_splits, y_train_splits, split_weights = split_training_data(X_train, y_train, K, iid, rng)    
    
    #Create K worker models
    workers = [model_constructor() for i in range(K)]
    
    #Create the global model
    global_model = model_constructor()
    global_weights = global_model.get_weights()
    
    #Execute the rounds of FederatedAveraging
    communication_rounds = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    m = int(np.ceil(C*K)) # Number of workers to use per round
    
    #Note: In the real world each worker would perform its update in parallel on a separate device.
    #      Here in simulation, we can perform worker updates sequentially on the same device.
    for t in range(MAX_T):
        #Randomly pick the workers to be used for this round
        worker_indices = rng.integers(K, size=m)
        
        #Perform the local update on each randomly selected worker starting from the global weights
        worker_weights = [global_weights for i in range(len(workers))]
        for wk_i in worker_indices:
            worker = workers[wk_i]
            worker.set_weights(global_weights)
            worker.fit(X_train_splits[wk_i], y_train_splits[wk_i], batch_size=B, epochs=E)
            #Return the worker weights to the "server"
            worker_weights[wk_i] = worker.get_weights()
            
        #Average all the worker weights to get the updated global weights
        for i in range(len(global_weights)):
            global_weights[i] = np.sum([split_weights[i]*w[i] for w in worker_weights], axis=0)
        
        #Evaluate the global model on the test set
        global_model.set_weights(global_weights)
        score = global_model.evaluate(X_train, y_train)
        communication_rounds["loss"].append(score[0])
        communication_rounds["accuracy"].append(score[1])
        val_score = global_model.evaluate(X_val, y_val)
        communication_rounds["val_loss"].append(val_score[0])
        communication_rounds["val_accuracy"].append(val_score[1])
    
    return global_model, communication_rounds


#Run experiments
experiments = [{"K": 100, "C": 0.1, "E": 1, "B": 10, "MAX_T": 50, "iid": True},
               {"K": 100, "C": 0.1, "E": 5, "B": 10, "MAX_T": 50, "iid": True}]
results = []
for experiment in experiments:
    _, communication_rounds = federated_averaging(X_train, y_train, X_test, y_test, rng=rng, **experiment)
    results.append((experiment, communication_rounds))

#Plot results
Target_Test_Accuracy = 0.99 # Measure number of rounds of communication to achieve this
fig, ax = plt.subplots()
for experiment, communication_rounds in results:
    ax.plot(np.arange(len(communication_rounds["val_accuracy"]))+1, 
                communication_rounds["val_accuracy"], 
                label="B={0} E={1}".format(experiment["B"], experiment["E"]))
xmax = max([len(communication_rounds["val_accuracy"]) for _, communication_rounds in results])
ax.hlines(Target_Test_Accuracy, xmin=1, xmax=xmax, colors="grey")
ax.set_xlabel("Communication Rounds")
ax.set_ylabel("Test Accuracy")
ax.set_title("MNIST CNN IID")
ax.legend()
plt.show()

############################################################################
# LASG (Chen, Tianyi, Yuejiao Sun, and Wotao Yin. 2020)
############################################################################
class lasg_worker(object):
    def __init__(self, model):
        self.model = model
        #the global weights at the last iteration that this worker uploaded
        self.last_upload_weights = None
        #list to keep track of the difference in global weight norms between each iteration and the previous one
        self.weights_diff_sq_norm_history = []
        self.staleness = 1
        self.X_train = None
        self.y_train = None
        
class lasg_server(object):
    def __init__(self, model_constructor, M):
        self.global_model = model_constructor()
        self.worker_gradients = [None] * M
        
def diff_global_sq_norm(a, b):
    diff = [t_a - t_b for t_a, t_b in zip(a, b)]
    return (tf.linalg.global_norm(diff)**2).numpy()
            
#------------------------------------------------------
#The following variables are named to match the paper.

M = 10  # M is the number of workers
K = 1000   # Maximum number of iterations of LASG to run
D = 50   # Maximum number of LASG iterations that a worker's stale gradient can be reused
eta = 0.05 # Step size
c = np.concatenate((np.zeros(D-10), np.array([(0.1/(eta**2))/(M**2)] * 10))) # Weights for RHS of LASG-WK2 condition

#The following variables are not assigned explicit names in the paper.

B = 0.01  # B is the worker minibatch size (integer > 0 or fraction of the worker local dataset size if B_is_fraction=True)
B_is_fraction = True
iid = True # If iid, we simulate an evenly distributed random split of the data across
            # workers. Otherwise each worker gets data in only one (or few) classes.
evaluation_interval = 25 # Evaluate global model whenever this many iterations of LASG have been run
#------------------------------------------------------

#Define the model, loss, and optimizer
def model_constructor():
    model = keras.Sequential([
            keras.Input(shape=X_train.shape[1:]),
            layers.Conv2D(32, kernel_size=(5, 5), activation="elu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(5, 5), activation="elu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dense(10, activation="softmax")
        ])
    
    sgd_optimizer = keras.optimizers.SGD(learning_rate=eta)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=sgd_optimizer, metrics=["accuracy"])
    
    return model

#Initialize the server
server = lasg_server(model_constructor, M)

#Initialize the workers
workers = [lasg_worker(model_constructor()) for i in range(M)]

#Partition the dataset into K splits and assign to workers
X_train_splits, y_train_splits, split_weights = split_training_data(X_train, y_train, M, iid, rng)
for i, worker in enumerate(workers):
    worker.X_train = X_train_splits[i]
    worker.y_train = y_train_splits[i]

#Execute the iterations of LASG and keep track of the number of communication rounds
iterations = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "iteration": [], "communication_rounds": []}
communication_rounds = 0

for k in range(K):
    #Broadcast the current global weights to each worker and use the LASG-WK2 condition to determine
    #which workers should upload their gradients
    global_weights = server.global_model.get_weights()
    for wk_i in range(M):
        worker = workers[wk_i]
        
        #Update the worker's weight difference square norm history (for use later in the LASG-WK2 condition)
        #On the first iteration all global and worker weights are randomized and the difference would be huge,
        #so we only measure the difference for subsequent iterations.
        if k > 0:
            worker.weights_diff_sq_norm_history.append(diff_global_sq_norm(global_weights, worker.model.get_weights()))
        
        #Get a randomly selected batch of size B from this worker's local data
        batch_size = B if not B_is_fraction else int(B*worker.X_train.shape[0])
        batch_indices = rng.integers(worker.X_train.shape[0], size=batch_size)
        X_train_batch = worker.X_train[batch_indices,]
        y_train_batch = worker.y_train[batch_indices,]
        
        #First, get the gradient of the loss on this batch w.r.t the worker's last upload weights.
        #(unless k == 0 or worker.staleness == D, in which case we don't bother because we will always upload)
        check_lasg_condition = k > 0 and worker.staleness < D
        if check_lasg_condition:
            worker.model.set_weights(worker.last_upload_weights)
            
            with tf.GradientTape() as tape:
                y_pred = worker.model(X_train_batch, training=True)
                loss = worker.model.compiled_loss(tf.convert_to_tensor(y_train_batch), 
                                            y_pred, regularization_losses=worker.model.losses)
            gradient_at_last_upload_weights = tape.gradient(loss, worker.model.trainable_variables)
            
        #Then get the gradient of the loss on this batch w.r.t. the current global weights.
        worker.model.set_weights(global_weights)
        
        with tf.GradientTape() as tape:
            y_pred = worker.model(X_train_batch, training=True)
            loss = worker.model.compiled_loss(tf.convert_to_tensor(y_train_batch), 
                                        y_pred, regularization_losses=worker.model.losses)
        gradient_at_current_weights = tape.gradient(loss, worker.model.trainable_variables)
        
        #Compute LASG-WK2 condition (if satisfied, we don't upload the new gradient)
        lasg_wk2_condition = False
        if check_lasg_condition:
            gradient_diff_sq_norm = diff_global_sq_norm(gradient_at_current_weights, gradient_at_last_upload_weights)
            weights_diff_sq_norm = np.array(worker.weights_diff_sq_norm_history[-D:]) @ c[-min(D, len(worker.weights_diff_sq_norm_history)):]
            
            lasg_wk2_condition = gradient_diff_sq_norm <= (1/(M**2))*weights_diff_sq_norm
            print ("{0:.8f} <= {1:.8f}: {2}".format(gradient_diff_sq_norm, (1/(M**2))*weights_diff_sq_norm, lasg_wk2_condition))
            
        if lasg_wk2_condition:
            #Increment the worker staleness
            worker.staleness += 1
        else:
            #"Upload" the new gradients and reset the staleness
            worker.last_upload_weights = worker.model.get_weights()
            server.worker_gradients[wk_i] = gradient_at_current_weights
            worker.staleness = 1
            communication_rounds += 1
        
    #Server updates (update global weights) with Generic LASG update rule
    trainable_vars = server.global_model.trainable_variables
    worker_gradients_sum = [tf.math.add_n([split_weights[i] * server.worker_gradients[i][j] for i in range(M)]) for j in range(len(trainable_vars))]
    server.global_model.optimizer.apply_gradients(zip(worker_gradients_sum, trainable_vars))
        
    #Evaluate the global model on the test set on the evaluation interval
    if k % evaluation_interval == 0:
        score = server.global_model.evaluate(X_train, y_train)
        iterations["loss"].append(score[0])
        iterations["accuracy"].append(score[1])
        val_score = server.global_model.evaluate(X_test, y_test)
        iterations["val_loss"].append(val_score[0])
        iterations["val_accuracy"].append(val_score[1])
        iterations["iteration"].append(k)
        iterations["communication_rounds"].append(communication_rounds)
        

fig, ax = plt.subplots()
ax.plot(iterations["communication_rounds"], iterations["val_accuracy"])
ax.set_xlabel("Communication (rounds of upload)")
ax.set_ylabel("Test Accuracy")
ax.set_title("LASG-WK2: MNIST CNN (IID)")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(iterations["iteration"], iterations["val_accuracy"])
ax.set_xlabel("Iteration")
ax.set_ylabel("Test Accuracy")
ax.set_title("LASG-WK2: MNIST CNN (IID)")
ax.legend()
plt.show()