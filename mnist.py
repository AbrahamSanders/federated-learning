"""
Script to compare performance of Vanilla SGD, FederatedAvergaging, and LASG-WK2 on
the MNIST dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

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
    rng = np.random.default_rng(seed=100)
    
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
                layers.Conv2D(32, kernel_size=(5, 5), activation="elu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(5, 5), activation="elu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dense(512, activation="elu"),
                layers.Dense(10, activation="softmax")
            ])
        
        sgd_optimizer = keras.optimizers.SGD(learning_rate=hparams.eta)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=sgd_optimizer, metrics=["accuracy"])
        
        return model
    
    ############################################################################
    # VANILLA SGD
    ############################################################################
    
    #Run experiment
    vanilla_sgd_hparams = vanilla_sgd.vanilla_sgd_hparams(epochs=4)
    model = model_constructor(vanilla_sgd_hparams)
    model.summary()
    hist = model.fit(X_train, y_train, 
                     batch_size=vanilla_sgd_hparams.batch_size, 
                     epochs=vanilla_sgd_hparams.epochs, 
                     validation_data=(X_test, y_test))
    
    batches_per_epoch = int(np.ceil(X_train.shape[0]/vanilla_sgd_hparams.batch_size))
    vanilla_sgd_iterations = (np.arange(len(hist.history["val_accuracy"]))+1) * batches_per_epoch
    
    ############################################################################
    # FEDERATED AVERAGING (McMahan, Brendan, et al. 2017)
    ############################################################################
    
    #Run experiment
    fedavg_hparams = federated_avg.fedavg_hparams()
    _, fedavg_iterations = federated_avg.federated_averaging(X_train, y_train, X_test, y_test, 
                                                             model_constructor, 
                                                             fedavg_hparams, rng)
    
    ############################################################################
    # LASG (Chen, Tianyi, Yuejiao Sun, and Wotao Yin. 2020)
    ############################################################################
    
    #Run experiment
    lasg_wk2_hparams = lasg.lasg_wk2_hparams()
    _, lasg_wk2_iterations = lasg.lasg_wk2(X_train, y_train, X_test, y_test, 
                                           model_constructor, 
                                           lasg_wk2_hparams, rng)
    
    
    ############################################################################
    # Plot results
    ############################################################################
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    axes[0].plot(vanilla_sgd_iterations, hist.history["loss"], label="Vanilla SGD")
    axes[0].plot(fedavg_iterations["iteration"], fedavg_iterations["loss"], label="FederatedAveraging")
    axes[0].plot(lasg_wk2_iterations["iteration"], lasg_wk2_iterations["loss"], label="LASG-WK2")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Vanilla SGD vs. FederatedAveraging vs. LASG-WK2 on MNIST CNN (IID)")
    axes[0].legend()
    axes[1].plot(fedavg_iterations["communication_rounds"], fedavg_iterations["loss"], label="FederatedAveraging")
    axes[1].plot(lasg_wk2_iterations["communication_rounds"], lasg_wk2_iterations["loss"], label="LASG-WK2")
    axes[1].set_xlabel("Communication (rounds of upload)")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("FederatedAveraging vs. LASG-WK2 on MNIST CNN (IID)")
    axes[1].legend()
    plt.show()
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    axes[0].plot(vanilla_sgd_iterations, hist.history["val_accuracy"], label="Vanilla SGD")
    axes[0].plot(fedavg_iterations["iteration"], fedavg_iterations["val_accuracy"], label="FederatedAveraging")
    axes[0].plot(lasg_wk2_iterations["iteration"], lasg_wk2_iterations["val_accuracy"], label="LASG-WK2")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("Vanilla SGD vs. FederatedAveraging vs. LASG-WK2 on MNIST CNN (IID)")
    axes[0].legend()
    axes[1].plot(fedavg_iterations["communication_rounds"], fedavg_iterations["val_accuracy"], label="FederatedAveraging")
    axes[1].plot(lasg_wk2_iterations["communication_rounds"], lasg_wk2_iterations["val_accuracy"], label="LASG-WK2")
    axes[1].set_xlabel("Communication (rounds of upload)")
    axes[1].set_ylabel("Test Accuracy")
    axes[1].set_title("FederatedAveraging vs. LASG-WK2 on MNIST CNN (IID)")
    axes[1].legend()
    plt.show()
    
if __name__ == "__main__":
    run()