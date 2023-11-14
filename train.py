import torch
from torch.optim import lr_scheduler
from torchinfo import summary
import pickle
import warnings
from itertools import chain

warnings.filterwarnings("ignore")

from data_setup import get_dataloaders
from model import ClassificationNetwork, RuleNetwork
from engine import train
from utils import save_model
from config import parse_arguments

args = parse_arguments()

# set hyperparameters for both networks
NUM_EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
PATIENCE = args.patience
BATCH_SIZE = args.batch_size

RULE_EMBED_DIM = args.rule_embed_dim
RULE_NUM_NODES = args.rule_num_hidden_nodes
RULE_NUM_LABELS = 43
RULE_MODEL_NAME = f"rule_lr{LEARNING_RATE}_bs{BATCH_SIZE}_ed{RULE_EMBED_DIM}_nhn{RULE_NUM_NODES}_pat{PATIENCE}"

CLASSIFICATION_EMBED_DIM = args.classification_embed_dim
CLASSIFICATION_NUM_NODES = args.classification_num_hidden_nodes
CLASSIFICATION_NUM_LABELS = 2
CLASSIFICATION_MODEL_NAME = f"classification_lr{LEARNING_RATE}_bs{BATCH_SIZE}_ed{CLASSIFICATION_EMBED_DIM}_nhn{CLASSIFICATION_NUM_NODES}_pat{PATIENCE}"

RANDOM_STATE = 42

# setting target device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# torch.set_default_device(device)

# setting seed
torch.cuda.manual_seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# embeddings = pickle.load(open("embeddings.pkl", "rb"))
# creating dataloaders, initialising model and the loss function

(
    train_dataloader,
    valid_dataloader,
    test_dataloader,
    vocab,
) = get_dataloaders("data.pkl", BATCH_SIZE)
classification_model = ClassificationNetwork(
    len(vocab),
    CLASSIFICATION_EMBED_DIM,
    CLASSIFICATION_NUM_NODES,
    CLASSIFICATION_NUM_LABELS,
)

rule_model = RuleNetwork(len(vocab), RULE_EMBED_DIM, RULE_NUM_NODES, RULE_NUM_LABELS)

# setting and optimizer
combined_parameters = chain(classification_model.parameters(), rule_model.parameters())
optimizer = torch.optim.Adam(combined_parameters, lr=LEARNING_RATE)

scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", patience=PATIENCE, verbose=True
)

# print(model)
# # printing model summary
# print("\nThe model summary is as follows: \n")
# summary(
#     model=model,
#     input_data=(
#         torch.rand(4000).type(torch.int64),
#         torch.cat(
#             (torch.tensor([0]), (torch.randint(1, 400, [BATCH_SIZE - 1]).sort()[0]))
#         ),
#     ),
#     col_names=["input_size", "output_size", "num_params", "trainable"],
#     col_width=20,
#     row_settings=["var_names"],
# )

print("\nStarting model training...")

# training and testing the model
results = train(
    classification_model=classification_model,
    rule_model=rule_model,
    train_dataloader=train_dataloader,
    valid_dataloader=valid_dataloader,
    optimizer=optimizer,
    epochs=NUM_EPOCHS,
    device=device,  # type:ignore
    scheduler=scheduler,
    classification_model_name=CLASSIFICATION_MODEL_NAME,
    rule_model_name=RULE_MODEL_NAME,
)

best_loss = min(results["test_loss"])
final_loss = results["test_loss"][-1]

# saving model results in pickle file
pickle.dump(
    results,
    open(
        f"results/bestloss{best_loss:.2f}_model_{RULE_MODEL_NAME}_{CLASSIFICATION_MODEL_NAME}_results.pkl",
        "wb",
    ),
    protocol=-1,
)

# saving the model
# save_model(
#     model,
#     f"{NETWORK}_finalloss{final_loss:.2f}_{MODEL_NAME}_final_model.pth",
# )

# make_prediction(pickle.load(open("data/samples.pkl","rb")),)
