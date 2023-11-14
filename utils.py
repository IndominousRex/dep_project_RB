import torch
import torch.utils.data
import torch.nn.functional as F
from pathlib import Path

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def save_model(model: torch.nn.Module, model_name: str):
    # Create target directory if it doesn't exist
    target_dir_path = Path("models")
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    model_save_path = target_dir_path / f"{model_name}"

    # Save the model state_dict()
    # print(f" Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

    print(f"Model saved at : {model_save_path}")


def load_model(model: torch.nn.Module, filename: str):
    # Create model save path
    target_dir_path = Path("models")
    model_save_path = target_dir_path / f"{filename}"

    # Loading the model state_dict()
    model.load_state_dict(torch.load(model_save_path))

    return model


# def make_prediction(
#     rules: List,
#     classification_model: torch.nn.Module,
#     rule_model: torch.nn.Module,
#     data_string: str,
#     data_tensor: torch.Tensor,
# ):
#     result = torch.tensor([0.0, 0.0])
#     num_rules_covering_data = 0

#     # combining the results of both models to get the final prediction
#     for i in range(len(rules)):
#         # compiling the regex
#         compiled_pattern = re.compile(rules[i], re.IGNORECASE)

#         rule_network_pred = torch.sigmoid(
#             (rule_model(data_tensor, torch.tensor([0])))[0][i]
#         )
#         # finding matching patterns
#         if bool(compiled_pattern.search(data_string)):
#             num_rules_covering_data += 1

#             # calculating result according to formula
#             if rule_network_pred > 0.5:
#                 result[1] += rule_network_pred.item()
#         else:
#             result[0] += 1 - rule_network_pred.item()

#     if num_rules_covering_data > 0:
#         result /= num_rules_covering_data

#     result[1] += 0.5 * (classification_model(data_tensor, torch.tensor([0])))[0][1]
#     result[0] += 0.5 * (classification_model(data_tensor, torch.tensor([0])))[0][0]

#     return torch.argmax(result).item()


def get_joint_score(
    one_hot_rule_classes,
    rule_network_probs,
    rule_coverage_matrix,
    classification_network_probs,
):
    rule_network_probs = rule_network_probs.unsqueeze(-1)  # (batch_size,num_rules,1)

    preds_mask = (rule_network_probs > 0.5).float()  # (batch_size,num_rules,1)

    # combining the results of both models to get the final prediction
    rule_coverage_matrix = rule_coverage_matrix.unsqueeze(
        -1
    )  # (batch_size,num_rules,1)
    rule_coverage_matrix_masked = (
        rule_coverage_matrix * preds_mask
    )  # (batch_size,num_rules,1)

    one_hot_rule_classes = one_hot_rule_classes.unsqueeze(0)  # (1,num_rules,2)

    rule_weight_product = rule_network_probs * one_hot_rule_classes + (
        1 - rule_network_probs
    ) * (
        1 - one_hot_rule_classes
    )  # (batch_size,num_rules,1)*(1,num_rules,2)=(batch_size,num_rules,2)

    sum_rule_firings = rule_coverage_matrix_masked.sum(1)  # (batch_size,)

    result = (
        rule_coverage_matrix_masked * rule_weight_product
    )  # (batch_size,num_rules,1)*(batch_size,num_rules,2)=(batch_size,num_rules,2)
    result = result.sum(dim=1) / (sum_rule_firings + 1e-20)  # (batch_size,2)
    result += classification_network_probs  # (batch_size,2)

    return result


def compute_LL_phi(
    rule_network_logits,
    rule_network_probs,
    rule_assigned_instance_labels,
    rule_coverage_matrix,
    labels,
    labelled_flag_matrix,
    rule_exemplar_matrix,
    num_rules,
):
    labels = labels.unsqueeze(-1).repeat((1, num_rules))  # (batch_size,num_rules)
    rule_labels_equals_true_labels = (rule_assigned_instance_labels == labels).to(
        torch.float32
    )  # (batch_size,num_rules)
    rule_labels_not_equals_true_labels = (rule_assigned_instance_labels != labels).to(
        torch.float32
    )  # (batch_size,num_rules)

    loss = F.binary_cross_entropy_with_logits(
        rule_network_logits, rule_labels_equals_true_labels, reduction="none"
    )  # (batch_size,num_rules)

    loss *= rule_coverage_matrix  # (batch_size,num_rules)
    loss = (rule_labels_not_equals_true_labels * loss) + (
        rule_labels_equals_true_labels * rule_exemplar_matrix * loss
    )  # (batch_size,num_rules)

    gcross_loss = generalized_cross_entropy_bernoulli(
        rule_network_probs, 0.2
    )  # (batch_size,num_rules)
    gcross_loss = (
        gcross_loss
        * rule_coverage_matrix
        * rule_labels_equals_true_labels
        * (1 - rule_exemplar_matrix)
    )  # (batch_size,num_rules)

    loss += gcross_loss  # (batch_size,num_rules)
    loss = torch.sum(loss, -1)  # (16)
    loss = loss * labelled_flag_matrix  # (16,16)
    loss = torch.mean(loss)  # ()

    return loss


def generalized_cross_entropy_loss(logits, targets, q=0.6):
    exp_logits = logits.exp()
    normalizer = torch.sum(exp_logits, -1, True)
    normalizer_q = torch.pow(normalizer, q)
    exp_logits_q = torch.exp(logits * q)
    f_j_q = exp_logits_q / normalizer_q
    loss = (1 - f_j_q) / q
    loss = torch.sum(loss * targets, -1)
    loss = torch.mean(loss)
    return loss


def generalized_cross_entropy_bernoulli(p, q=0.2):
    return (1 - torch.pow(p, q)) / q


def compute_implication_loss(
    rule_network_probs,
    classification_network_probs,
    rule_coverage_matrix,
    rule_classes,
    num_classes,
    labelled_flag_matrix,
):
    psi = 1e-25  # a small value to avoid nans

    one_hot_mask = (
        F.one_hot(rule_classes, num_classes).float().to(device)
    )  # (num_rules,2)

    classification_network_probs = classification_network_probs @ one_hot_mask.T
    # (batch_size,2) @ (2,num_rules) = (batch_size,num_rules)
    obj = 1 - (
        rule_network_probs
        * (1 - classification_network_probs)  # (batch_size,num_rules)
    )  # (Argument of log in equation 4)

    # computing last term of equation 5, will multiply with gamma outside this function
    obj = rule_coverage_matrix * torch.log(obj + psi)  # (batch_size,num_rules)
    obj = torch.sum(obj, -1)  # (16)
    obj = obj * (
        1 - labelled_flag_matrix
    )  # defined only for instances in U, so mask by (1-d) # (batch_size,batch_size)

    obj = torch.mean(obj)  # ()
    return -obj


def compute_loss(
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
):
    labels %= num_classes
    labels_one_hot = F.one_hot(labels, num_classes).float()  # (batch_size,2)
    LL_theta = F.binary_cross_entropy_with_logits(
        classification_network_logits, labels_one_hot, reduction="none"
    )
    LL_theta = (labelled_flag_matrix * LL_theta).mean()

    LL_phi = compute_LL_phi(
        rule_network_logits,
        rule_network_probs,
        rule_assigned_instance_labels,
        rule_coverage_matrix,
        labels,
        labelled_flag_matrix,
        rule_exemplar_matrix,
        num_rules,
    )

    implication_loss = compute_implication_loss(
        rule_network_probs,
        classification_network_probs,
        rule_coverage_matrix,
        rule_labels,
        num_classes,
        labelled_flag_matrix,
    )

    # print(
    #     f"LL_phi shape = {LL_phi.shape}, LL_theta shape = {LL_theta.shape}, implication_loss shape = {implication_loss.shape}"
    # )
    # print(
    #     f"\nLL_phi = {LL_phi}, LL_theta = {LL_theta}, implication_loss = {implication_loss}"
    # )

    loss = LL_phi + LL_theta + gamma * implication_loss

    return loss
