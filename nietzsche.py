"""
Script to compare performance of Vanilla SGD, FederatedAvergaging, and LASG-WK2 on
the Nietzsche corpus.
Character-LSTM implementation follows this example at
https://keras.io/examples/generative/lstm_character_level_text_generation/
"""

import io
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

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

    path = keras.utils.get_file(
    "nietzsche.txt", origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt"
    )
    with io.open(path, encoding="utf-8") as f:
        text = f.read().lower()
    text = text.replace("\n", " ")  # We remove newlines chars for nicer display
    print("Corpus length:", len(text))
    
    chars = sorted(list(set(text)))
    print("Total chars:", len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    
    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 40
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i : i + maxlen])
        next_chars.append(text[i + maxlen])
    print("Number of sequences:", len(sentences))
    
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences)), dtype=np.int32)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i] = char_indices[next_chars[i]]
        
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=100)
    
    #Define the model, loss, and optimizer
    def model_constructor(hparams):
        model = keras.Sequential([
                keras.Input(shape=(maxlen, len(chars))),
                layers.LSTM(128),
                layers.Dense(len(chars), activation="softmax"),
            ])
        
        sgd_optimizer = keras.optimizers.SGD(learning_rate=hparams.eta)
        model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                      optimizer=sgd_optimizer, metrics=["accuracy"])
        
        return model
    
    ############################################################################
    # RUN EXPERIMENTS
    ############################################################################
    
    #Run each experiment until this target test accuracy is reached
    target_test_accuracy = 0.17
    
    vanilla_sgd_hparams = vanilla_sgd.vanilla_sgd_hparams(eta=1.5, batch_size=32, epochs=100, evaluation_interval=5, 
                                                          target_val_accuracy=target_test_accuracy)
    
    #The hyperparameter variables for the federated learning algorithms are set to align with their respective papers.
    #This can cause confusion: for example the hparam 'K' means # of workers for FederatedAveraging and # of iterations for LASG.
    #See the constructor docstrings for a reference on what each hyperparameter does for each algorithm.
    
    fedavg_hparams = federated_avg.fedavg_hparams(K=100, C=0.1, E=1, B=10, eta=1.5, MAX_T=10000, evaluation_interval=2, 
                                                  target_val_accuracy=target_test_accuracy)
    
    lasg_wk2_hparams = lasg.lasg_wk2_hparams(M=10, K=10000, D=50, c=0.1, c_range=10, eta=1.5, B=100, evaluation_interval=5, 
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
                        target_test_accuracy, iid=True, dataset="Nietzsche", modeltype="LSTM", mode="save")
    
    runner.plot_results(non_iid_vanilla_sgd_log, non_iid_fedavg_log, non_iid_lasg_wk2_log, 
                        target_test_accuracy, iid=False, dataset="Nietzsche", modeltype="LSTM", mode="save")
    
if __name__ == "__main__":
    run()