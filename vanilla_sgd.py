"""
Support for vanilla SGD using the same pattern
as the federated learning algorithms.
"""

import numpy as np
import utils
from tensorflow import keras

class vanilla_sgd_hparams(object):
    """
    Hyperparameters for vanilla SGD
    """
    def __init__(self, eta=0.1, batch_size=32, epochs=100, iid=True, evaluation_interval=5, target_val_accuracy=0.9):
        """
        Parameters
        ----------
        eta : float, optional
            Step size to use for SGD optimizer. The default is 0.1.
        batch_size : int, optional
            Minibatch size. The default is 32.
        epochs : int, optional
           Number of epochs to train for. The default is 100.
        iid : boolean, optional
            If iid, the training set is shuffled randomly. Otherwise it is sorted by target.
            The default is True.
        evaluation_interval : int, optional
            Evaluate model whenever this many iterations of SGD have been run. The default is 5.
        target_val_accuracy : float, optional
            Stop training if this target validation accuracy has been achieved. The default is 0.9.
        """
        self.eta = eta
        self.batch_size=batch_size
        self.epochs=epochs
        self.iid = iid
        self.evaluation_interval = evaluation_interval
        self.target_val_accuracy = target_val_accuracy


def sgd(X_train, y_train, X_val, y_val, model_constructor, hparams, rng=None):
    """
    Train a model using vanilla SGD on a single device.
    Return the final model and metrics gathered over the course of the run.
    
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
    hparams : vanilla_sgd_hparams
        Hyperparameters for SGD.
    rng : numpy.random.Generator, optional
       instance to use for random number generation.

    Returns
    -------
    model : tf.keras.Model
        The final model
    
    log : dict
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
    """
    if rng is None:
        rng = np.random.default_rng()
    
    #Initialize the model
    model = model_constructor(hparams)
    
    #Shuffle or sort the training set (depending on hparams.iid)
    num_batches = int(np.ceil(X_train.shape[0] / hparams.batch_size))
    X_train_splits, y_train_splits, _ = utils.split_training_data(X_train, y_train, num_batches, hparams.iid, rng)
    X_train = np.concatenate(X_train_splits)
    y_train = np.concatenate(y_train_splits)
    
    #Do initial evaluation of the randomly initialized global model as a baseline
    log = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "iteration": []}
    utils.evaluate_and_log(log, model, X_train, y_train, X_val, y_val, 0)
    
    def on_batch_end(batch, logs):
        if not model.stop_training:
            iteration = model.optimizer.iterations.numpy()
            #Keras only evaluates on validation data at the end of each epoch, and we want this
            #to happen more frequently (at the specified evaluation interval).
            if iteration % hparams.evaluation_interval == 0:
                #Evaluate on the validation set
                utils.evaluate_and_log(log, model, X_val=X_val, y_val=y_val, iteration=iteration)
                #Reuse the batch-wise training set metric estimations given by Keras
                log["loss"].append(logs["loss"])
                log["accuracy"].append(logs["accuracy"])
                
                #Stop training when we have reached the target validation accuracy
                if log["val_accuracy"][-1] >= hparams.target_val_accuracy:
                    model.stop_training = True
    
    model.fit(X_train, y_train, 
              batch_size=hparams.batch_size, 
              epochs=hparams.epochs,
              shuffle=False,
              callbacks=[keras.callbacks.LambdaCallback(on_batch_end=on_batch_end)])
    
    return model, log