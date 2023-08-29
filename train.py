import torch
from torch import nn
from torch.optim import lr_scheduler
from torchinfo import summary
import pickle
import warnings

warnings.filterwarnings("ignore")

from data_setup import get_dataloaders
from model import ClassificationNetwork, RuleNetwork
from engine import train
from utils import save_model
from config import parse_arguments

args = parse_arguments()

# set hyperparameters
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs
EMBED_DIM = args.embed_dim
NUM_NODES = args.num_hidden_nodes
NETWORK = args.network_name
NUM_LABELS = 43 if NETWORK == "rule" else 2
MODEL_NAME = args.model_name
RANDOM_STATE = 12345
# setting target device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# setting seed
torch.cuda.manual_seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# embeddings = pickle.load(open("embeddings.pkl", "rb"))
# creating dataloaders, initialising model and the loss function
if NETWORK == "classification":
    train_dataloader, test_dataloader, vocab_len = get_dataloaders(
        "data.pkl", BATCH_SIZE
    )
    model = ClassificationNetwork(vocab_len, EMBED_DIM, NUM_NODES, NUM_LABELS, None)
    loss_fn = nn.CrossEntropyLoss()

else:
    train_dataloader, test_dataloader, vocab_len = get_dataloaders(
        "rule_data.pkl", BATCH_SIZE
    )
    model = RuleNetwork(vocab_len, EMBED_DIM, NUM_NODES, NUM_LABELS)
    loss_fn = nn.BCEWithLogitsLoss()

# setting and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# scheduler = lr_scheduler.StepLR(optimizer)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=2, verbose=True)

# print(model)
# printing model summary
print("\nThe model summary is as follows: \n")
summary(
    model=model,
    input_data=(
        torch.rand(4000).type(torch.int64),
        torch.randint(0, 400, [16]).sort()[0],
    ),
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"],
)

print("\nStarting model training...")

# training and testing the model
results = train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS,
    device=device,  # type: ignore
    scheduler=scheduler,
    network=NETWORK,
    model_name=MODEL_NAME,
)

# saving model results in pickle file
pickle.dump(results, open(f"results/{MODEL_NAME}_results.pkl", "wb"), protocol=-1)

# # saving the model
# save_model(model=model, model_name=f"{MODEL_NAME}.pth")
