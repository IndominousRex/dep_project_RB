import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_name",
        help="name of the model to be saved",
        type=str,
    )
    parser.add_argument(
        "--learning_rate", help="learning rate of the model", type=float, default=3e-4
    )
    parser.add_argument(
        "--batch_size", help="batch size for training", type=int, default=16
    )
    parser.add_argument(
        "--epochs",
        help="number of epochs to run the training for",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--embed_dim",
        help="size of the embedding vector for each token",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--num_hidden_nodes",
        help="number of nodes in the hidden layers of the network",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--network_name",
        help="name of the network to be trained",
        default="rule",
        choices=["rule", "classification"],
        type=str,
    )

    args = parser.parse_args()

    return args
