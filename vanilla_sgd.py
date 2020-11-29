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
    def __init__(self, eta=0.1, batch_size=32, epochs=100, evaluation_interval=5, target_val_accuracy=0.9):
        """
        Parameters
        ----------
        eta : float, optional
            Step size to use for SGD optimizer. The default is 0.1.
        batch_size : int, optional
            Minibatch size. The default is 32.
        epochs : int, optional
           Number of epochs to train for. The default is 100.
        evaluation_interval : int, optional
            Evaluate model whenever this many iterations of SGD have been run. The default is 5.
        target_val_accuracy : float, optional
            Stop training if this target validation accuracy has been achieved. The default is 0.9.
        """
        self.eta = eta
        self.batch_size=batch_size
        self.epochs=epochs
        self.evaluation_interval = evaluation_interval
        self.target_val_accuracy = target_val_accuracy


def sgd(X_train, y_train, X_val, y_val, model_constructor, hparams):
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
    
    #Initialize the model
    model = model_constructor(hparams)
    
    #Do initial evaluation of the randomly initialized global model as a baseline
    log = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "iteration": []}
    utils.evaluate_and_log(log, model, X_train, y_train, X_val, y_val, 0)
    
    #We set the steps_per_epoch parameter to hparams.evaluation_interval to force keras to evaluate the 
    #model when we want - otherwise it will wait until the end of a full training set epoch.
    #the user doesn't know this, so we need to scale up hparams.epochs by the number of evaluation intervals
    #in a full training set epoch.
    
    evaluation_intervals_per_epoch = X_train.shape[0] / (hparams.evaluation_interval * hparams.batch_size)
    actual_epochs = int(np.ceil(hparams.epochs * evaluation_intervals_per_epoch))
    
    def on_epoch_end(epoch, logs):
        #Stop training when we have reached the target validation accuracy
        if logs["val_accuracy"] >= hparams.target_val_accuracy:
            model.stop_training = True
    
    hist = model.fit(X_train, y_train, 
                     batch_size=hparams.batch_size, 
                     epochs=actual_epochs,
                     validation_data=(X_val, y_val), 
                     steps_per_epoch=hparams.evaluation_interval,
                     shuffle=False,
                     callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)])
    
    log["loss"].extend(hist.history["loss"])
    log["accuracy"].extend(hist.history["accuracy"])
    log["val_loss"].extend(hist.history["val_loss"])
    log["val_accuracy"].extend(hist.history["val_accuracy"])
    log["iteration"].extend(((np.arange(len(hist.history["loss"]))+1)*hparams.evaluation_interval).tolist())
    
    return model, log