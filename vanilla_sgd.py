"""
Support for vanilla SGD (using Keras directly) using the same pattern
as the federated learning algorithms.
"""

class vanilla_sgd_hparams(object):
    """
    Hyperparameters for vanilla SGD
    """
    def __init__(self, eta=0.1, batch_size=32, epochs=10):
        """
        Parameters
        ----------
        eta : float, optional
            Step size to use for SGD optimizer. The default is 0.1.
        batch_size : int, optional
            Minibatch size. The default is 32.
        epochs : int, optional
           Number of epochs to train for. The default is 10.  
        """
        self.eta = eta
        self.batch_size=batch_size
        self.epochs=epochs