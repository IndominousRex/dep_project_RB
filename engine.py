import torch
from tqdm.auto import tqdm
from utils import get_joint_score, compute_loss, save_model

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
)
import torchmetrics.classification.f_beta
import torchmetrics.classification.accuracy
import torchmetrics.classification.precision_recall
import torch.utils.data


# def train_step(
#     model: torch.nn.Module,
#     dataloader: torch.utils.data.DataLoader,
#     loss_fn: torch.nn.Module,
#     optimizer: torch.optim.Optimizer,
#     device: torch.device,
#     acc_fn: torchmetrics.classification.accuracy.BinaryAccuracy,
#     prec_fn: torchmetrics.classification.precision_recall.BinaryPrecision,
#     recall_fn: torchmetrics.classification.precision_recall.BinaryRecall,
#     f1_fn: torchmetrics.classification.f_beta.BinaryF1Score,
#     network="rule",
# ):
#     # Put model in train mode
#     model.train()

#     # Setup train metrics
#     train_loss, train_acc, train_precision, train_recall, train_f1 = 0, 0, 0, 0, 0

#     # Loop through data loader data batches
#     for batch, (y, X, offsets) in enumerate(dataloader):
#         # Send data to target device
#         X, offsets = X.to(device), offsets.to(device)
#         y = y.to(device) if network == "classification" else y.to(device).float()

#         # 1. Forward pass
#         train_pred_logits = model(X, offsets)

#         # 2. Calculating and accumulating loss
#         loss = loss_fn(train_pred_logits, y)
#         train_loss += loss.item()

#         # 3. Optimizer zero grad
#         optimizer.zero_grad()

#         # 4. Loss backward
#         loss.backward()

#         # 5. Optimizer step
#         optimizer.step()

#         # Generating the predicted values from the logits
#         # argmax is used for the binary classifier to be used along with nn.CrossEntropyLoss
#         # sigmoid is used for the multi label prediction of the rule network to be used along with nn.BCEWithLogitsLoss
#         if network == "classification":
#             train_pred = torch.argmax(train_pred_logits, dim=1)
#         else:
#             train_pred = torch.sigmoid(train_pred_logits)
#             train_pred[train_pred >= 0.5] = 1
#             train_pred[train_pred < 0.5] = 0

#         # aggregating the metrics for the batch
#         train_acc += acc_fn(train_pred, y).cpu().numpy()
#         train_precision += prec_fn(train_pred, y).cpu().numpy()
#         train_recall += recall_fn(train_pred, y).cpu().numpy()
#         train_f1 += f1_fn(train_pred, y).cpu().numpy()

#     # Adjust metrics to get average
#     train_loss /= len(dataloader)
#     train_acc /= len(dataloader)
#     train_precision /= len(dataloader)
#     train_recall /= len(dataloader)
#     train_f1 /= len(dataloader)

#     return train_loss, train_acc, train_precision, train_recall, train_f1


def train_step(
    classification_model: torch.nn.Module,
    rule_model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    acc_fn: torchmetrics.classification.accuracy.BinaryAccuracy,
    prec_fn: torchmetrics.classification.precision_recall.BinaryPrecision,
    recall_fn: torchmetrics.classification.precision_recall.BinaryRecall,
    f1_fn: torchmetrics.classification.f_beta.BinaryF1Score,
):
    # Put model in train mode
    classification_model.train()
    rule_model.train()

    # Setup train metrics
    train_loss, train_acc, train_precision, train_recall, train_f1 = 0, 0, 0, 0, 0

    # Loop through data loader data batches
    for _, (
        labels,
        X,
        offsets,
        rule_assigned_instance_labels,
        rule_coverage_matrix,
        labelled_flag_matrix,
        rule_exemplar_matrix,
    ) in enumerate(dataloader):
        # Send data to target device
        (
            labels,
            X,
            offsets,
            rule_assigned_instance_labels,
            rule_coverage_matrix,
            labelled_flag_matrix,
            rule_exemplar_matrix,
        ) = (
            labels.to(device),
            X.to(device),
            offsets.to(device),
            rule_assigned_instance_labels.to(device),
            rule_coverage_matrix.to(device),
            labelled_flag_matrix.to(device),
            rule_exemplar_matrix.to(device),
        )

        num_classes = 2
        num_rules = 43
        gamma = 0.1

        rule_labels = torch.ones((num_rules), dtype=torch.int64)
        one_hot_rule_classes = (
            torch.nn.functional.one_hot(rule_labels, num_classes).float().to(device)
        )

        # 1. Forward pass
        rule_network_logits = rule_model(X, offsets)
        rule_network_probs = rule_network_logits.sigmoid()
        classification_network_logits = classification_model(X, offsets)
        classification_network_probs = classification_network_logits.sigmoid()

        joint_score = get_joint_score(
            one_hot_rule_classes,
            rule_network_probs,
            rule_coverage_matrix,
            classification_network_probs,
        )
        train_pred = joint_score.argmax(1)

        train_loss = compute_loss(
            num_classes,
            rule_labels,
            rule_network_logits,
            rule_network_probs,
            classification_network_logits,
            classification_network_probs,
            rule_assigned_instance_labels,
            rule_coverage_matrix,
            labelled_flag_matrix,
            rule_exemplar_matrix,
            labels,
            num_rules,
            gamma,
        )

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        train_loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # aggregating the metrics for the batch
        train_acc += acc_fn(train_pred, labels).cpu().numpy()
        train_precision += prec_fn(train_pred, labels).cpu().numpy()
        train_recall += recall_fn(train_pred, labels).cpu().numpy()
        train_f1 += f1_fn(train_pred, labels).cpu().numpy()

    # Adjust metrics to get average
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    train_precision /= len(dataloader)
    train_recall /= len(dataloader)
    train_f1 /= len(dataloader)

    return train_loss, train_acc, train_precision, train_recall, train_f1


def test_step(
    classification_model: torch.nn.Module,
    rule_model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    acc_fn: torchmetrics.classification.accuracy.BinaryAccuracy,
    prec_fn: torchmetrics.classification.precision_recall.BinaryPrecision,
    recall_fn: torchmetrics.classification.precision_recall.BinaryRecall,
    f1_fn: torchmetrics.classification.f_beta.BinaryF1Score,
):
    # Put model in eval mode
    classification_model.eval()
    rule_model.eval()

    # Setup test metrics
    test_loss, test_acc, test_precision, test_recall, test_f1 = 0, 0, 0, 0, 0

    # Using inference mode the disable gradient backpropagaton
    with torch.inference_mode():
        # Loop through DataLoader batches
        for _, (
            labels,
            X,
            offsets,
            rule_assigned_instance_labels,
            rule_coverage_matrix,
            labelled_flag_matrix,
            rule_exemplar_matrix,
        ) in enumerate(dataloader):
            # Send data to target device
            (
                labels,
                X,
                offsets,
                rule_assigned_instance_labels,
                rule_coverage_matrix,
                labelled_flag_matrix,
                rule_exemplar_matrix,
            ) = (
                labels.to(device),
                X.to(device),
                offsets.to(device),
                rule_assigned_instance_labels.to(device),
                rule_coverage_matrix.to(device),
                labelled_flag_matrix.to(device),
                rule_exemplar_matrix.to(device),
            )

            num_classes = 2
            num_rules = 43
            gamma = 0.1

            rule_labels = torch.ones((num_rules), dtype=torch.int64)
            one_hot_rule_classes = (
                torch.nn.functional.one_hot(rule_labels, num_classes).float().to(device)
            )

            # 1. Forward pass
            rule_network_logits = rule_model(X, offsets)
            rule_network_probs = rule_network_logits.sigmoid()
            classification_network_logits = classification_model(X, offsets)
            classification_network_probs = classification_network_logits.sigmoid()

            joint_score = get_joint_score(
                one_hot_rule_classes,
                rule_network_probs,
                rule_coverage_matrix,
                classification_network_probs,
            )
            test_pred = joint_score.argmax(1)

            test_loss = compute_loss(
                num_classes,
                rule_labels,
                rule_network_probs,
                rule_network_logits,
                classification_network_logits,
                classification_network_probs,
                rule_assigned_instance_labels,
                rule_coverage_matrix,
                labelled_flag_matrix,
                rule_exemplar_matrix,
                labels,
                num_rules,
                gamma,
            )

            test_acc += acc_fn(test_pred, labels).cpu().numpy()
            test_precision += prec_fn(test_pred, labels).cpu().numpy()
            test_recall += recall_fn(test_pred, labels).cpu().numpy()
            test_f1 += f1_fn(test_pred, labels).cpu().numpy()

        # Adjust metrics to get average loss and accuracy per batch
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        test_precision /= len(dataloader)
        test_recall /= len(dataloader)
        test_f1 /= len(dataloader)

    return test_loss, test_acc, test_precision, test_recall, test_f1


def train(
    classification_model: torch.nn.Module,
    rule_model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    rule_model_name: str,
    classification_model_name: str,
):
    # initialising functions to calaculate the model metrics
    acc_fn = BinaryAccuracy().to(device)
    prec_fn = BinaryPrecision().to(device)
    recall_fn = BinaryRecall().to(device)
    f1_fn = BinaryF1Score().to(device)

    # Create empty results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "test_loss": [],
        "test_acc": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": [],
    }

    # Make sure model is on target device
    classification_model.to(device)
    rule_model.to(device)

    best_loss = torch.inf
    best_epoch = 0
    early_stop_threshold = 10

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, train_precision, train_recall, train_f1 = train_step(
            classification_model,
            rule_model,
            train_dataloader,
            optimizer,
            device,
            acc_fn,
            prec_fn,
            recall_fn,
            f1_fn,
        )

        test_loss, test_acc, test_precision, test_recall, test_f1 = test_step(
            classification_model,
            rule_model,
            valid_dataloader,
            device,
            acc_fn,
            prec_fn,
            recall_fn,
            f1_fn,
        )

        scheduler.step(test_loss)

        if test_loss < best_loss:
            best_epoch = epoch
            best_loss = test_loss
            save_model(
                classification_model,
                f"classification_{classification_model_name}_best_model.pth",
            )
            save_model(
                rule_model,
                f"rule_{rule_model_name}_best_model.pth",
            )

        elif epoch - best_epoch > early_stop_threshold:
            print(f"\nEarly stopped training at epoch: {epoch}")
            print(f"Best test loss : {best_loss:.4f} at epoch: {best_epoch}")
            break

        # Print out what's happening
        print(
            f"\nEpoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"train_prec: {train_precision:.4f} | "
            f"train_recall: {train_recall:.4f} | "
            f"train_f1: {train_f1:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f} | "
            f"test_prec: {test_precision:.4f} | "
            f"test_recall: {test_recall:.4f} | "
            f"test_f1: {test_f1:.4f} | \n"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_precision"].append(train_precision)
        results["train_recall"].append(train_recall)
        results["train_f1"].append(train_f1)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["test_precision"].append(test_precision)
        results["test_recall"].append(test_recall)
        results["test_f1"].append(test_f1)

    # Return the calculated results at the end of the epochs or after early stopping
    return results
