"""
Implementation of simulator for FederatedAveraging as described in the paper
Communication-Efficient Learning of Deep Networks from Decentralized Data
(McMahan, Brendan, et al. 2017)
http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf
"""

import numpy as np
import utils

class fedavg_worker(object):
    """
    Class to represent a FederatedAveraging worker
    (called a "Client" in the paper)
    """
    def __init__(self, model):
        """
        Initialize the worker with a Keras model

        Parameters
        ----------
        model : tf.keras.Model
            The model to be used by the worker to do local training.
        """
        self.model = model
        self.X_train = None
        self.y_train = None
        
class fedavg_server(object):
    """
    Class to represent a FederatedAveraging parameter server
    """
    def __init__(self, model, K):
        """
        Initialize the parameter server with a Keras model
        and a list of size K to store K worker weights

        Parameters
        ----------
        model : tf.keras.Model
            The model to be used by the parameter server as the global model.
        K : int
            The number of workers that can participate in federated learning.
        """
        self.global_model = model
        self.worker_weights = [None] * K
        
        
class fedavg_hparams(object):
    """
    Hyperparameters for the FederatedAveraging algorithm.
    """
    def __init__(self, K=100, C=0.1, E=1, B=10, eta=0.1, MAX_T=1000, iid=True, evaluation_interval=2):
        """
        Parameters
        ----------
        -----------------------------------------------------------
        The following variables are named to match the paper:
            
        K : int, optional
            The number of workers. The default is 100.
        C : float, optional
            The fraction of workers randomly selected per iteration. The default is 0.1.
        E : int, optional
            The number of epochs a worker will run over its data per iteration. The default is 1.
        B : int, optional
            The worker minibatch size. The default is 10.
        eta : float, optional
            Step size to use for worker SGD optimizers. The default is 0.1.
        -----------------------------------------------------------
        -----------------------------------------------------------
        The following variables are not assigned explicit names in the paper:
            
        MAX_T : int, optional
            Maximum number of rounds of FederatedAveraging to run. The default is 1000.
        iid : boolean, optional
            If iid, we simulate an evenly distributed random split of the data across
            workers. Otherwise each worker gets data in only one (or few) classes. 
            The default is True.
        evaluation_interval : int, optional
            Evaluate global model whenever this many iterations of FederatedAveraging have been run. The default is 2.
        -----------------------------------------------------------
        """
        self.K = K
        self.C = C
        self.E = E
        self.B = B
        self.eta = eta
        self.MAX_T = MAX_T
        self.iid = iid
        self.evaluation_interval = evaluation_interval

def federated_averaging(X_train, y_train, X_val, y_val, model_constructor, hparams, rng=None):
    """
    Simulate training a model using FederatedAveraging across K distributed devices.
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
    hparams : fedavg_hparams
        Hyperparameters for FederatedAveraging.
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
    server = fedavg_server(model_constructor(hparams), hparams.K)
    
    #Initialize the workers
    workers = [fedavg_worker(model_constructor(hparams)) for i in range(hparams.K)]
    
    #Partition the dataset into K splits and assign to workers
    
    #Note: In the real world we would not have access to the dataset as it would be distributed
    #      across all the worker devices. Here in simulation, we have access to the complete dataset
    #      and define the splits that go to each worker.
    
    X_train_splits, y_train_splits, split_weights = utils.split_training_data(X_train, y_train, hparams.K, hparams.iid, rng)    
    for i, worker in enumerate(workers):
        worker.X_train = X_train_splits[i]
        worker.y_train = y_train_splits[i]

    #Execute the iterations of FederatedAveraging and keep track of the number of communication rounds
    iterations = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "iteration": [], "communication_rounds": []}
    communication_rounds = 0
    
    m = int(np.ceil(hparams.C * hparams.K)) # Number of workers to use per iteration

    #Note: In the real world each worker would perform its update in parallel on a separate device.
    #      Here in simulation, we can perform worker updates sequentially on the same device.
    for t in range(hparams.MAX_T):
        global_weights = server.global_model.get_weights()
        #Randomly pick the workers to be used for this iteration
        worker_indices = set(rng.integers(hparams.K, size=m))
        
        #Perform the local update on each randomly selected worker starting from the global weights
        for wk_i in worker_indices:
            worker = workers[wk_i]
            worker.model.set_weights(global_weights)
            worker.model.fit(worker.X_train, worker.y_train, batch_size=hparams.B, epochs=hparams.E)
            #Upload the worker weights to the server
            server.worker_weights[wk_i] = worker.model.get_weights()
        communication_rounds += m
            
        #Average all the worker weights to get the updated global weights
        for i in range(len(global_weights)):
            global_weights[i] = np.sum(
                [split_weights[wk_i]*(server.worker_weights[wk_i][i] if wk_i in worker_indices else global_weights[i]) for wk_i in range(hparams.K)], 
                axis=0)
        server.global_model.set_weights(global_weights)
        
        #Evaluate the global model on the test set on the evaluation interval
        if (t+1) % hparams.evaluation_interval == 0:
            score = server.global_model.evaluate(X_train, y_train)
            iterations["loss"].append(score[0])
            iterations["accuracy"].append(score[1])
            val_score = server.global_model.evaluate(X_val, y_val)
            iterations["val_loss"].append(val_score[0])
            iterations["val_accuracy"].append(val_score[1])
            iterations["iteration"].append(t+1)
            iterations["communication_rounds"].append(communication_rounds)
    
    return server.global_model, iterations