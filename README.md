# FashionMNIST CNN in PyTorch

A small CNN (TinyVGG-style: 2 conv blocks + linear classifier) trained on FashionMNIST.
The architecture and `helper_functions.py` follow the computer vision chapter of [learnpytorch.io](https://www.learnpytorch.io).

Checkpoint selected on a 5k validation split; test evaluated once at the end.
Results table and plots coming in the next commit.

## My deep learning path

1. Andrew Ng's Machine Learning Specialization: linear and logistic regression, gradient descent, L2 regularization.
2. From scratch in NumPy: sigmoid, cost, gradient, gradient descent, predict.
3. PyTorch fundamentals and classification: [pytorch-classification-basics](https://github.com/Salman-Elhaddadi/pytorch-classification-basics).
4. Computer vision / CNNs: this repo.

## Run

    pip install -r requirements.txt
    python train.py --hidden-units 10
    python train.py --hidden-units 64

Optional flags: `--lr` (default 0.01), `--epochs` (default 10).

## Credits

Architecture and helper functions: [learnpytorch.io](https://www.learnpytorch.io) by Daniel Bourke.
