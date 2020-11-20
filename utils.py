"""
Utility methods shared by all federated learning simulations
"""

import numpy as np
import tensorflow as tf

def split_training_data(X_train, y_train, num_splits, iid, rng):
    """
    Splits training data into partitions that can be distributed among
    workers.

    Parameters
    ----------
    X_train : numpy ndarray
        Training features.
    y_train : numpy ndarray
        Training targets.
    num_splits : int
        Number of partiitons to create.
    iid : boolean
        True to uniformly distribute the examples and targets across the splits.
        False to order the examples by target before creating the splits.
    rng : numpy.random.Generator
       instance to use for random number generation.

    Returns
    -------
    X_train_splits : list of numpy ndarray
        The training feature splits.
    y_train_splits : list of numpy ndarray
        The training target splits.
    split_weights : list of float
        The proportion of the total training set present in each split

    """
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

def diff_global_sq_norm(a, b):
    """
    Compute the square of the global norm of the difference of two lists of tensors.
    Each tensor in b is subtracted from each tensor in a entrywise, and then
    tf.linalg.global_norm is used on the resulting list of difference tensors to get the norm.
    The square of this quantity is returned.

    Parameters
    ----------
    a : list of tf.Tensor
        
    b : list of tf.Tensor
        
    Returns
    -------
    float
        The square of the global norm of the difference between a and b.
    """
    diff = [t_a - t_b for t_a, t_b in zip(a, b)]
    return (tf.linalg.global_norm(diff)**2).numpy()