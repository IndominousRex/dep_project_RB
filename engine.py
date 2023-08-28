import torch
from tqdm.auto import tqdm

# from sklearn.metrics import f1_score
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,  # type: ignore
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    f1_fn,
    acc_fn,
):
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_f1, train_acc = 0, 0, 0

    # Loop through data loader data batches
    for batch, (y, X, offsets) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        train_pred_logits = model(X, offsets)

        # 2. Calculating and accumulating loss
        loss = loss_fn(train_pred_logits, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Generating the predicted values from the logits
        y_pred = torch.argmax(train_pred_logits, dim=1)

        # y_pred = torch.sigmoid(y_pred_logits)
        # y_pred[y_pred >= 0.5] = 1
        # y_pred[y_pred < 0.5] = 1

        train_f1 += f1_fn(y_pred, y).cpu().numpy()
        train_acc += acc_fn(y_pred, y).cpu().numpy()
        # temp = y - y_pred_logits
        # c = 0
        # for i in temp.reshape(-1):
        #     if i.abs() < 0.5:
        #         c += 1
        # batch_acc += c / num_classes

        # train_acc += c / len(y) / len(y[1])

        # train_acc += (y_pred_class == y).sum().item() / len(y_pred)
        # print(y_pred, y_pred_class, train_prec, "\n")
        # break

    # Adjust metrics to get average loss and accuracy per batch
    train_loss /= len(dataloader)
    train_f1 /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_f1, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,  # type: ignore
    loss_fn: torch.nn.Module,
    device: torch.device,
    f1_fn,
    acc_fn,
):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_f1, test_acc = 0, 0, 0
    # y_true, y_pred = [], []
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (y, X, offsets) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X, offsets)

            # 2. Calculate and accumulate loss

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            # test_pred = torch.sigmoid(test_pred_logits)
            # test_pred[test_pred >= 0.5] = 1
            # test_pred[test_pred < 0.5] = 1

            test_pred = torch.argmax(test_pred_logits, dim=1)

            test_f1 += f1_fn(test_pred, y).cpu().numpy()
            test_acc += acc_fn(test_pred, y).cpu().numpy()
            # temp = y - test_pred
            # c = 0
            # for i in temp.reshape(-1):
            #     if i.abs() < 0.5:
            #         c += 1
            #     # batch_acc += c / num_classes

            # test_acc += c / len(y) / len(y[1])

            # break

    # Adjust metrics to get average loss and accuracy per batch
    test_loss /= len(dataloader)
    test_f1 = test_f1 / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_f1, test_acc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,  # type: ignore
    test_dataloader: torch.utils.data.DataLoader,  # type: ignore
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    num_labels: int,
):
    # initialising functions tp calaculate the model metrics
    f1_fn = BinaryF1Score().to(device)
    acc_fn = BinaryAccuracy().to(device)

    # Create empty results dictionary
    results = {
        "train_loss": [],
        "train_f1": [],
        "train_acc": [],
        "test_loss": [],
        "test_f1": [],
        "test_acc": [],
    }

    # Make sure model on target device
    model.to(device)

    best_test_loss = float("inf")
    patience = 5  # Number of epochs to wait before decreasing learning rate
    early_stop_counter = 0

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_f1, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            f1_fn=f1_fn,
            acc_fn=acc_fn,
        )
        test_loss, test_f1, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
            f1_fn=f1_fn,
            acc_fn=acc_fn,
        )

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.1  # Reduce learning rate by a factor of 10
                print(f"Lr reduced to {param_group['lr']}\n")
            early_stop_counter = 0  # Reset early stopping counter

        # Print out what's happening
        print(
            "\n"
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_f1: {train_f1:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_f1: {test_f1:.4f} | "
            f"test_acc: {test_acc:.4f} |"
        )

        if early_stop_counter >= patience:
            print("Early stopping due to increasing test loss.")
            break

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_f1"].append(train_f1)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_f1"].append(test_f1)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results
