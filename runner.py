"""
Utility methods to run several federated learning simulation experiments at once
and plot the results.
"""

import numpy as np
import matplotlib.pyplot as plt

import lasg
import federated_avg
import vanilla_sgd

def run_experiments(X_train, y_train, X_test, y_test, model_constructor, 
                    vanilla_sgd_hparams=None, fedavg_hparams=None, lasg_wk2_hparams=None, seed=None):
    """
    Runs experiments on the provided data using a model built by the provided model constructor.
    
    Experiments are done using Vanilla SGD, FederatedAveraging, and LASG-WK2.

    Parameters
    ----------
    X_train : numpy ndarray
        Training features.
    y_train : numpy ndarray
        Training targets.
    X_test : numpy ndarray
        Test features.
    y_test : numpy ndarray
        Test targets.
    model_constructor : function
        function that constructs a compiled tf.keras.Model using hparams.
    vanilla_sgd_hparams : vanilla_sgd.vanilla_sgd_hparams
        Hyperparameters for Vanilla SGD.
    fedavg_hparams : federated_avg.fedavg_hparams
        Hyperparameters for FederatedAveraging.
    lasg_wk2_hparams : lasg.lasg_wk2_hparams
        Hyperparameters for LASG-WK2.
    seed : int, optional
       seed to use for random number generation.

    Returns
    -------
    vanilla_sgd_log : dict of lists
        A logging container object for vanilla SGD, containing the experiment results.
        
    fedavg_log : dict of lists
        A logging container object for FederatedAveraging, containing the experiment results.
        
    lasg_wk2_log : dict of lists
        A logging container object for LASG-WK2, containing the experiment results.
    """
    
    ############################################################################
    # VANILLA SGD
    ############################################################################
    
    #Run experiment
    if vanilla_sgd_hparams is not None:
        print()
        print("Running Experiment: VANILLA SGD")
        print()
        rng = np.random.default_rng(seed)
        _, vanilla_sgd_log = vanilla_sgd.sgd(X_train, y_train, X_test, y_test,
                                             model_constructor,
                                             vanilla_sgd_hparams, rng)
    else:
        vanilla_sgd_log = None
    
    ############################################################################
    # FEDERATED AVERAGING (McMahan, Brendan, et al. 2017)
    ############################################################################
    
    #Run experiment
    if fedavg_hparams is not None:
        print()
        print("Running Experiment: FEDERATED AVERAGING")
        print()
        rng = np.random.default_rng(seed)
        _, fedavg_log = federated_avg.federated_averaging(X_train, y_train, X_test, y_test,
                                                          model_constructor,
                                                          fedavg_hparams, rng)
    else:
        fedavg_log = None
    
    ############################################################################
    # LASG (Chen, Tianyi, Yuejiao Sun, and Wotao Yin. 2020)
    ############################################################################
    
    #Run experiment
    if lasg_wk2_hparams is not None:
        print()
        print("Running Experiment: LASG")
        print()
        rng = np.random.default_rng(seed)
        _, lasg_wk2_log = lasg.lasg_wk2(X_train, y_train, X_test, y_test,
                                        model_constructor,
                                        lasg_wk2_hparams, rng)
    else:
        lasg_wk2_log = None
    
    return vanilla_sgd_log, fedavg_log, lasg_wk2_log

def plot_results(vanilla_sgd_log, fedavg_log, lasg_wk2_log, target_test_accuracy, iid):
    """
    Plots results comparing vanilla SGD, FederatedAveraging, and LASG-WK2.

    Parameters
    ----------
    vanilla_sgd_log : dict of lists
        A logging container object for vanilla SGD.
    fedavg_log : dict of lists
        A logging container object for FederatedAveraging.
    lasg_wk2_log : dict of lists
        A logging container object for LASG-WK2.
    target_test_accuracy : float
        The target test accuracy for the experiments, to show on the plots.
    iid : boolean
        Indicates if experiments were performed with iid data.
 
    Returns
    -------
    None.

    """
    iid_label = "IID" if iid else "Non-IID"
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    axes[0].plot(vanilla_sgd_log["iteration"], vanilla_sgd_log["loss"], color="black", label="Vanilla SGD")
    axes[0].plot(fedavg_log["iteration"], fedavg_log["loss"], label="FederatedAveraging")
    axes[0].plot(lasg_wk2_log["iteration"], lasg_wk2_log["loss"], label="LASG-WK2")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Vanilla SGD vs. FederatedAveraging vs. LASG-WK2 \n on MNIST CNN ({0})".format(iid_label))
    axes[0].legend()
    axes[1].plot(fedavg_log["communication_rounds"], fedavg_log["loss"], label="FederatedAveraging")
    axes[1].plot(lasg_wk2_log["communication_rounds"], lasg_wk2_log["loss"], label="LASG-WK2")
    axes[1].set_xlabel("Communication (rounds of upload)")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("FederatedAveraging vs. LASG-WK2 \n on MNIST CNN ({0})".format(iid_label))
    axes[1].legend()
    plt.show()
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    axes[0].plot(vanilla_sgd_log["iteration"], vanilla_sgd_log["val_accuracy"], color="black", label="Vanilla SGD")
    axes[0].plot(fedavg_log["iteration"], fedavg_log["val_accuracy"], label="FederatedAveraging")
    axes[0].plot(lasg_wk2_log["iteration"], lasg_wk2_log["val_accuracy"], label="LASG-WK2")
    axes[0].axhline(target_test_accuracy, color="grey", linestyle="--", label="Target Accuracy ({0})".format(target_test_accuracy))
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("Vanilla SGD vs. FederatedAveraging vs. LASG-WK2 \n on MNIST CNN ({0})".format(iid_label))
    axes[0].legend()
    axes[1].plot(fedavg_log["communication_rounds"], fedavg_log["val_accuracy"], label="FederatedAveraging")
    axes[1].plot(lasg_wk2_log["communication_rounds"], lasg_wk2_log["val_accuracy"], label="LASG-WK2")
    axes[1].axhline(target_test_accuracy, color="grey", linestyle="--", label="Target Accuracy ({0})".format(target_test_accuracy))
    axes[1].set_xlabel("Communication (rounds of upload)")
    axes[1].set_ylabel("Test Accuracy")
    axes[1].set_title("FederatedAveraging vs. LASG-WK2 \n on MNIST CNN ({0})".format(iid_label))
    axes[1].legend()
    plt.show()
    
    avg_fedavg_upload_fraction = np.mean(fedavg_log["worker_upload_fraction"][1:])
    avg_lasg_wk2_upload_fraction = np.mean(lasg_wk2_log["worker_upload_fraction"][1:])
    fig, ax = plt.subplots()
    ax.bar(fedavg_log["iteration"], fedavg_log["worker_upload_fraction"], label="FederatedAveraging")
    ax.bar(lasg_wk2_log["iteration"], lasg_wk2_log["worker_upload_fraction"], label="LASG-WK2")
    ax.axhline(avg_fedavg_upload_fraction, color="tab:blue", linestyle="--", label="FederatedAveraging (Avg.): {0:.2f}".format(avg_fedavg_upload_fraction))
    ax.axhline(avg_lasg_wk2_upload_fraction, color="tab:orange", linestyle="--", label="LASG-WK2 (Avg.): {0:.2f}".format(avg_lasg_wk2_upload_fraction))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Avg. Fraction of workers (uploads)")
    ax.set_title("FederatedAveraging vs. LASG-WK2 \n on MNIST CNN ({0})".format(iid_label))
    ax.legend()
    plt.show()