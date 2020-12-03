"""
Utility methods shared by all federated learning simulations
"""

import numpy as np
import tensorflow as tf

def split_training_data(X_train, y_train, num_splits, iid, rng, shards_per_split=2):
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
        False to order the examples by target before creating the splits, and then distribute
        the examples unevenly in quantity across the splits in a manner where each split only receives
        examples belonging to a small number of target labels.
    rng : numpy.random.Generator
       instance to use for random number generation.
    shards_per_split : int
        For non-I.I.D splits, the technique of McMahan, et al. is used:
        after ordering the examples by target, the data is divided into 
        shards_per_split * num_splits shards, and each split is allocated shards_per_split 
        randomly selected shards. In the paper, shards_per_split=2.
        If iid=True, this parameter does nothing. The default is 2.

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
        shards_per_split = 1
        train_order = rng.permutation(X_train.shape[0])
    else:
        train_order = np.argsort(y_train)
        
    X_train = X_train[train_order]
    y_train = y_train[train_order]
    
    X_train_shards = np.array_split(X_train, shards_per_split * num_splits)
    y_train_shards = np.array_split(y_train, shards_per_split * num_splits)
    
    if not iid:
        #Randomly concatenate shards_per_split number of shards to create each split.
        X_train_shards = np.array(X_train_shards)
        y_train_shards = np.array(y_train_shards)
        shards_order = rng.permutation(len(X_train_shards))
        X_train_splits = []
        y_train_splits = []
        for i in range(num_splits):
            begin = shards_per_split*i
            end = begin + shards_per_split
            shard_allocation = shards_order[begin:end]
            X_train_splits.append(np.concatenate(X_train_shards[shard_allocation]))
            y_train_splits.append(np.concatenate(y_train_shards[shard_allocation]))
            
        #If iid=False, we also need to randomize the split sizes. We can do this by looping through each split
        #and re-allocating a random number of examples to the next split (up to 80% of the current split).
        for i in range(num_splits-1):
            max_examples_to_move = int(X_train_splits[i].shape[0] * 0.8)
            examples_to_move = rng.integers(max_examples_to_move)
            if examples_to_move > 0:
                X_train_splits[i+1] = np.concatenate((X_train_splits[i][-examples_to_move:,], X_train_splits[i+1]))
                y_train_splits[i+1] = np.concatenate((y_train_splits[i][-examples_to_move:,], y_train_splits[i+1]))
                X_train_splits[i] = X_train_splits[i][:-examples_to_move,]
                y_train_splits[i] = y_train_splits[i][:-examples_to_move,]
    else:
        X_train_splits = X_train_shards
        y_train_splits = y_train_shards
            

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

def evaluate_and_log(log, model, X_train=None, y_train=None, X_val=None, y_val=None, iteration=None, 
                     communication_rounds=None, num_total_workers=None):
    """
    Evaluate a model and log the metrics in the provided logging container object.

    Parameters
    ----------
    log : dict of lists
        A logging container object.
    model : tf.keras.Model
        The model to evaluate
    X_train : numpy ndarray, optional
        Training features, if applicable.
    y_train : numpy ndarray, optional
        Training targets, if applicable.
    X_val : numpy ndarray, optional
        Validation features, if applicable.
    y_val : numpy ndarray, optional
        Validation targets, if applicable.
    iteration : int, optional
        The current optimization step, if applicable.
    communication_rounds : int, optional
        The current number of communication rounds, if applicable.
    num_total_workers : int, optional
        The total number of workers, if applicable.

    Returns
    -------
    None.

    """
    if X_train is not None and y_train is not None:
        score = model.evaluate(X_train, y_train)
        log["loss"].append(score[0])
        log["accuracy"].append(score[1])
        
    if X_val is not None and y_val is not None:
        val_score = model.evaluate(X_val, y_val)
        log["val_loss"].append(val_score[0])
        log["val_accuracy"].append(val_score[1])
        
    if iteration is not None:
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