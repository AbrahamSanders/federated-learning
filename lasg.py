"""
Implementation of simulator for LASG-WK2 as described in the paper
LASG: Lazily Aggregated Stochastic Gradients for Communication-Efficient Distributed Learning
(Chen, Tianyi, Yuejiao Sun, and Wotao Yin. 2020)
https://arxiv.org/abs/2002.11360
"""

import numpy as np
import tensorflow as tf
import utils

class lasg_worker(object):
    """
    Class to represent a LASG worker
    """
    def __init__(self, model):
        """
        Initialize the worker with a Keras model

        Parameters
        ----------
        model : tf.keras.Model
            The model to be used by the worker to compute gradients.
        """
        self.model = model
        #the global weights at the last iteration that this worker uploaded
        self.last_upload_weights = None
        #list to keep track of the difference in global weight norms between each iteration and the previous one
        self.weights_diff_sq_norm_history = []
        self.staleness = 1
        self.X_train = None
        self.y_train = None
        
class lasg_server(object):
    """
    Class to represent a LASG parameter server
    """
    def __init__(self, model, M):
        """
        Initialize the parameter server with a Keras model
        and a list of size M to store M worker gradients

        Parameters
        ----------
        model : tf.keras.Model
            The model to be used by the parameter server as the global model.
        M : int
            The number of workers that can participate in federated learning.
        """
        self.global_model = model
        self.worker_gradients = [None] * M


class lasg_wk2_hparams(object):
    """
    Hyperparameters for the LASG-WK2 algorithm.
    """
    def __init__(self, M=10, K=1000, D=50, c=None, eta=0.05, B=0.01, B_is_fraction=True, iid=True, evaluation_interval=25):
        """
        Parameters
        ----------
        -----------------------------------------------------------
        The following variables are named to match the paper:
        
        M : int, optional
            The number of workers. The default is 10.
        K : int, optional
            Maximum number of iterations of LASG to run. The default is 1000.
        D : int, optional
            Maximum number of LASG iterations that a worker's stale gradient can be reused. The default is 50.
        c : 1-d numpy ndarray of float. The default is computed from D, eta, and M as specified in the paper.
            Weights for RHS of LASG-WK2 condition
        eta : float, optional
            Step size to use for worker SGD optimizers. The default is 0.05.
        -----------------------------------------------------------
        -----------------------------------------------------------
        The following variables are not assigned explicit names in the paper:
            
        B : int or float, optional
            The worker minibatch size (integer > 0 or fraction of the worker local dataset size if B_is_fraction=True). 
            The default is 0.01.
        B_is_fraction : boolean, optional
            See B above. The default is True.
        iid : boolean, optional
            If iid, we simulate an evenly distributed random split of the data across
            workers. Otherwise each worker gets data in only one (or few) classes. 
            The default is True.
        evaluation_interval : int, optional
            Evaluate global model whenever this many iterations of LASG have been run. The default is 25.
        -----------------------------------------------------------
        """
        self.M = M
        self.K = K
        self.D = D
        self.c = c if c is not None else np.concatenate((np.zeros(D-10), np.array([(0.1/(eta**2))/(M**2)] * 10)))
        self.eta = eta
        self.B = B
        self.B_is_fraction = B_is_fraction
        self.iid = iid
        self.evaluation_interval = evaluation_interval

def lasg_wk2(X_train, y_train, X_val, y_val, model_constructor, hparams, rng=None):
    """
    Simulate training a model using LASG-WK2 across M distributed devices.
    Return the final global model and metrics gathered over the course of the run.

    Parameters
    ----------
    X_train : numpy ndarray
        Training features.
    y_train : numpy ndarray
        Training targets.
    X_val : numpy ndarray
        Validation features.
    y_val : numpy ndarray
        Validation targets.
    model_constructor : function
        function that constructs a compiled tf.keras.Model using hparams.
    hparams : lasg_wk2_hparams
        Hyperparameters for LASG-WK2.
    rng : numpy.random.Generator, optional
       instance to use for random number generation.

    Returns
    -------
    global_model : tf.keras.Model
        The final global model
    
    iterations : dict
        Dictionary containing training and validation metrics:
            loss : 
                training loss at each iteration
            accuracy : 
                training accuracy at each iteration
            val_loss : 
                validation loss at each iteration
            val_accuracy : 
                validation accuracy at each iteration
            iteration : 
                the iteration number at which the measurements were made
            communication_rounds : 
                the cumulative number of worker uploads by each iteration
    """
    if rng is None:
        rng = np.random.default_rng()
    
    #Initialize the server
    server = lasg_server(model_constructor(hparams), hparams.M)
    
    #Initialize the workers
    workers = [lasg_worker(model_constructor(hparams)) for i in range(hparams.M)]
    
    #Partition the dataset into M splits and assign to workers
    
    #Note: In the real world we would not have access to the dataset as it would be distributed
    #      across all the worker devices. Here in simulation, we have access to the complete dataset
    #      and define the splits that go to each worker.
    
    X_train_splits, y_train_splits, split_weights = utils.split_training_data(X_train, y_train, hparams.M, hparams.iid, rng)
    for i, worker in enumerate(workers):
        worker.X_train = X_train_splits[i]
        worker.y_train = y_train_splits[i]
    
    #Execute the iterations of LASG and keep track of the number of communication rounds
    iterations = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "iteration": [], "communication_rounds": []}
    communication_rounds = 0
    
    #Note: In the real world each worker would perform its update in parallel on a separate device.
    #      Here in simulation, we can perform worker updates sequentially on the same device.
    for k in range(hparams.K):
        #Broadcast the current global weights to each worker and use the LASG-WK2 condition to determine
        #which workers should upload their gradients
        global_weights = server.global_model.get_weights()
        for wk_i in range(hparams.M):
            worker = workers[wk_i]
            
            #Update the worker's weight difference square norm history (for use later in the LASG-WK2 condition)
            #On the first iteration all global and worker weights are randomized and the difference would be huge,
            #so we only measure the difference for subsequent iterations.
            if k > 0:
                worker.weights_diff_sq_norm_history.append(utils.diff_global_sq_norm(global_weights, worker.model.get_weights()))
            
            #Get a randomly selected batch of size B from this worker's local data
            batch_size = hparams.B if not hparams.B_is_fraction else int(hparams.B*worker.X_train.shape[0])
            batch_indices = rng.integers(worker.X_train.shape[0], size=batch_size)
            X_train_batch = worker.X_train[batch_indices,]
            y_train_batch = worker.y_train[batch_indices,]
            
            #First, get the gradient of the loss on this batch w.r.t the worker's last upload weights.
            #(unless k == 0 or worker.staleness == D, in which case we don't bother because we will always upload)
            check_lasg_condition = k > 0 and worker.staleness < hparams.D
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
                gradient_diff_sq_norm = utils.diff_global_sq_norm(gradient_at_current_weights, gradient_at_last_upload_weights)
                weights_diff_sq_norm = np.array(worker.weights_diff_sq_norm_history[-hparams.D:]) @ hparams.c[-min(hparams.D, len(worker.weights_diff_sq_norm_history)):]
                
                lasg_wk2_condition = gradient_diff_sq_norm <= (1/(hparams.M**2))*weights_diff_sq_norm
                print ("{0:.8f} <= {1:.8f}: {2}".format(gradient_diff_sq_norm, (1/(hparams.M**2))*weights_diff_sq_norm, lasg_wk2_condition))
                
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
        worker_gradients_sum = [tf.math.add_n([split_weights[i] * server.worker_gradients[i][j] for i in range(hparams.M)]) for j in range(len(trainable_vars))]
        server.global_model.optimizer.apply_gradients(zip(worker_gradients_sum, trainable_vars))
            
        #Evaluate the global model on the test set on the evaluation interval
        if (k+1) % hparams.evaluation_interval == 0:
            score = server.global_model.evaluate(X_train, y_train)
            iterations["loss"].append(score[0])
            iterations["accuracy"].append(score[1])
            val_score = server.global_model.evaluate(X_val, y_val)
            iterations["val_loss"].append(val_score[0])
            iterations["val_accuracy"].append(val_score[1])
            iterations["iteration"].append(k+1)
            iterations["communication_rounds"].append(communication_rounds)
            
    return server.global_model, iterations