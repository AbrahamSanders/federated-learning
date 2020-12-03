# federated-learning
A Keras implementation of [FederatedAvergaing](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf) [McMahan, Brendan, et al. 2017] and [Lazily Aggregated Stochastic Gradients](https://arxiv.org/pdf/2002.11360.pdf) [Chen, Tianyi, Yuejiao Sun, and Wotao Yin. 2020]

## Dependencies
Tensorflow 2.3 or greater


## Directions
1. Set desired hyperparameters for Vanilla SGD, FederatedAveraging, and LASG-WK2 within mnist.py or nietzsche.py
2. Execute the script
```shell
python mnist.py
```
or
```shell
python nietzsche.py
```
3. When the experiments complete, results will be saved to the current working directory.
