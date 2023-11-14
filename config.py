import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        "-e",
        help="number of epochs to run the training for (default: 100)",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        help="learning rate of the model (default: 3e-4)",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--patience",
        "-p",
        help="Number of epochs with no improvement after which learning rate will be reduced (default: 5)",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        help="batch size for training (default: 16)",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--rule_embed_dim",
        "-red",
        help="size of the embedding vector for each token in the rule model (default: 256)",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--rule_num_hidden_nodes",
        "-rnhn",
        help="number of nodes in the hidden layers of the rule model (default: [512,512])",
        type=int,
        nargs="+",
        default=[512, 512],
    )

    parser.add_argument(
        "--classification_embed_dim",
        "-ced",
        help="size of the embedding vector for each token in the classification model (default: 256)",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--classification_num_hidden_nodes",
        "-cnhn",
        help="number of nodes in the hidden layers of the classification model (default: 256)",
        type=int,
        default=256,
    )

    args = parser.parse_args()

    return args
