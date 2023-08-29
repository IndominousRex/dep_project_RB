import torch
from pathlib import Path


def save_model(model: torch.nn.Module, model_name: str):
    # Create target directory if it doesn't exist
    target_dir_path = Path("models")
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    # print(f" Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def load_model(model: torch.nn.Module, filename: str):
    # Create model save path
    target_dir_path = Path("models")
    model_save_path = target_dir_path / filename

    # Loading the model state_dict()
    model.load_state_dict(torch.load(model_save_path))
