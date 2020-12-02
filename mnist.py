"""
Script to compare performance of Vanilla SGD, FederatedAvergaging, and LASG-WK2 on
the MNIST dataset.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

import runner
import lasg
import federated_avg
import vanilla_sgd

def run():
    """
    Execute the script
    """
    ############################################################################
    # SETUP
    ############################################################################

    #Load the data
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    #Scale the images
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255
    
    #Add black/white channel dimension
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    
    #Define the model, loss, and optimizer
    def model_constructor(hparams):
        model = keras.Sequential([
                keras.Input(shape=X_train.shape[1:]),
                layers.Conv2D(20, kernel_size=(5, 5), activation="elu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(50, kernel_size=(5, 5), activation="elu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dense(500, activation="elu"),
                layers.Dense(10, activation="linear")
            ])
        
        sgd_optimizer = keras.optimizers.SGD(learning_rate=hparams.eta)
        model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                      optimizer=sgd_optimizer, metrics=["accuracy"])
        
        return model
    
    ############################################################################
    # RUN EXPERIMENTS
    ############################################################################
    
    #Run each experiment until this target test accuracy is reached
    target_test_accuracy = 0.95
    
    vanilla_sgd_hparams = vanilla_sgd.vanilla_sgd_hparams(eta=0.1, batch_size=32, epochs=100, evaluation_interval=5, 
                                                          target_val_accuracy=target_test_accuracy)
    
    #The hyperparameter variables for the federated learning algorithms are set to align with their respective papers.
    #This can cause confusion: for example the hparam 'K' means # of workers for FederatedAveraging and # of iterations for LASG.
    #See the constructor docstrings for a reference on what each hyperparameter does for each algorithm.
    
    fedavg_hparams = federated_avg.fedavg_hparams(K=100, C=0.1, E=1, B=10, eta=0.1, MAX_T=10000, evaluation_interval=2, 
                                                  target_val_accuracy=target_test_accuracy)
    
    lasg_wk2_hparams = lasg.lasg_wk2_hparams(M=10, K=10000, D=50, c=4000, c_range=10, eta=0.1, B=100, evaluation_interval=5, 
                                             target_val_accuracy=target_test_accuracy, 
                                             print_lasg_wk2_condition=True)
    
    #Run with the training set distributed I.I.D
    print()
    print("Running Experiments with training set distributed I.I.D")
    print()
    
    vanilla_sgd_hparams.iid = True
    fedavg_hparams.iid = True
    lasg_wk2_hparams.iid = True
    
    iid_vanilla_sgd_log, iid_fedavg_log, iid_lasg_wk2_log = runner.run_experiments(
                    X_train, y_train, X_test, y_test, model_constructor, 
                    vanilla_sgd_hparams, fedavg_hparams, lasg_wk2_hparams, seed=100)
    
    #Run with the training set distributed non-I.I.D
    #We use the the technique of McMahan, et al. to split the data among workers in a
    #"pathologically" non-IID manner where each worker may only have a few labels represented
    #in its local data. Additionally, we also randomize the volume of data each worker receives.
    print()
    print("Running Experiments with training set distributed non-I.I.D")
    print()
    
    vanilla_sgd_hparams.iid = False
    fedavg_hparams.iid = False
    lasg_wk2_hparams.iid = False
    
    #The stepsize needs to be lowered in the non-IID setting to maintain
    #stable convergence.
    vanilla_sgd_hparams.eta = 0.07
    fedavg_hparams.eta = 0.07
    lasg_wk2_hparams.eta = 0.07
    
    non_iid_vanilla_sgd_log, non_iid_fedavg_log, non_iid_lasg_wk2_log = runner.run_experiments(
                    X_train, y_train, X_test, y_test, model_constructor, 
                    vanilla_sgd_hparams, fedavg_hparams, lasg_wk2_hparams, seed=100)
    
    print()
    print("Done!")
    print()
    
    ############################################################################
    # Plot results
    ############################################################################
    runner.plot_results(iid_vanilla_sgd_log, iid_fedavg_log, iid_lasg_wk2_log, 
                        target_test_accuracy, iid=True, dataset="MNIST", modeltype="CNN", mode="save")
    
    runner.plot_results(non_iid_vanilla_sgd_log, non_iid_fedavg_log, non_iid_lasg_wk2_log, 
                        target_test_accuracy, iid=False, dataset="MNIST", modeltype="CNN", mode="save")
    
if __name__ == "__main__":
    run()