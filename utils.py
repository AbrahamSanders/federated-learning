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
    
    if not iid:
        #If iid=False, sorting the data by the labels is not enough.
        #We also need to randomize the partition sizes. We can do this by looping through each partition
        #and re-allocating a random number of examples to the next partition (up to 90% of the current partition).
        for i in range(num_splits-1):
            max_examples_to_move = int(X_train_splits[i].shape[0] * 0.9)
            examples_to_move = rng.integers(max_examples_to_move)
            X_train_splits[i+1] = np.concatenate((X_train_splits[i][-examples_to_move:,], X_train_splits[i+1]))
            y_train_splits[i+1] = np.concatenate((y_train_splits[i][-examples_to_move:,], y_train_splits[i+1]))
            X_train_splits[i] = X_train_splits[i][:-examples_to_move,]
            y_train_splits[i] = y_train_splits[i][:-examples_to_move,]
            

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

def evaluate_and_log(log, model, X_train, y_train, X_val, y_val, iteration, communication_rounds=None, num_total_workers=None):
    """
    Evaluate a model and log the metrics in the provided logging container object.

    Parameters
    ----------
    log : dict of lists
        A logging container object.
    model : tf.keras.Model
        The model to evaluate
    X_train : numpy ndarray
        Training features.
    y_train : numpy ndarray
        Training targets.
    X_val : numpy ndarray
        Validation features.
    y_val : numpy ndarray
        Validation targets.
    iteration : int
        the current optimization step.
    communication_rounds : int, optional
        the current number of communication rounds, if applicable.
    num_total_workers : int, optional
        the total number of workers, if applicable.

    Returns
    -------
    None.

    """
    score = model.evaluate(X_train, y_train)
    log["loss"].append(score[0])
    log["accuracy"].append(score[1])
    val_score = model.evaluate(X_val, y_val)
    log["val_loss"].append(val_score[0])
    log["val_accuracy"].append(val_score[1])
    log["iteration"].append(iteration)
    
    if communication_rounds is not None:
        log["communication_rounds"].append(communication_rounds)
    
        if num_total_workers is not None:
            if iteration > 0:
                last_iteration = 0 if len(log["iteration"]) == 1 else log["iteration"][-2]
                evaluation_interval = iteration - last_iteration
                last_comm_rounds = 0 if len(log["communication_rounds"]) == 1 else log["communication_rounds"][-2]
                new_comm_rounds = communication_rounds - last_comm_rounds
                log["worker_upload_fraction"].append(new_comm_rounds / (evaluation_interval * num_total_workers))
            else:
                log["worker_upload_fraction"].append(0.0)