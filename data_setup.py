import torch
import random
import pickle
import re
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader

RANDOM_STATE = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(RANDOM_STATE)
# torch.set_default_device(device)

vocab = tokenizer = None
rules = pickle.load(open("data/rules.pkl", "rb"))
exemplars = pickle.load(open("data/exemplars.pkl", "rb"))
num_classes = 2


# function to preprocess the text by removing links, numbers, hashtags, symbols and emoticons if any
def preprocess(text):
    text = re.sub(r"@[A-Za-z0-9]+", "", text)
    text = re.sub(r"[0-9]+", "", text)
    text = re.sub(r"#[A-Za-z0-9]*", "", text)
    text = re.sub(r"https?:\/\/\S+", "", text)
    text = re.sub(r"[\n]", " ", text)
    text = re.sub(r"['\",*%$#()]", " ", text)
    # text = re.sub(r"[.]", " . ", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text.strip()


# generator function to return tokens (to be used with build_vocab_from_iterator)
def yield_tokens(data_iter, tokenizer):
    for text, _ in data_iter:
        yield tokenizer(text)


# creating the vocab object from the data using the spacy tokenizer
def make_vocab(data):
    tokenizer = get_tokenizer("spacy")

    data = [[preprocess(i[0]), i[1]] for i in data]

    vocab = build_vocab_from_iterator(yield_tokens(data, tokenizer), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    return data, vocab, tokenizer


# converting a token or a list of tokens into numbers
def stoi(x):
    global vocab, tokenizer
    return vocab(tokenizer(x))  # type: ignore


# function to create the input tensor and the offsets from a input batch
def collate_batch(batch):
    batch_size = len(batch)
    label_list, text_list, offsets = [], [], [0]
    rule_assigned_instance_labels = torch.zeros(batch_size, len(rules))
    rule_exemplar_matrix = torch.zeros(batch_size, len(rules))
    rule_coverage_matrix = torch.zeros(batch_size, len(rules))
    labelled_flag_matrix = torch.zeros((batch_size))

    for i in range(batch_size):
        text = batch[i][0]
        label = batch[i][1]

        if label != num_classes:
            labelled_flag_matrix[i] = 1

        for j in range(len(rules)):
            # compiling the regex
            compiled_pattern = re.compile(rules[j], re.IGNORECASE)

            # making sparse matrix where 1 denotes the ith datapoint is an exemplar for rule j
            if text == preprocess(exemplars[j]):
                rule_exemplar_matrix[i][j] = 1

            # finding matching patterns
            if bool(compiled_pattern.search(text)):
                rule_assigned_instance_labels[i][j] = 1
                rule_coverage_matrix[i][j] = 1
            else:
                rule_assigned_instance_labels[i][j] = num_classes

        label_list.append(label)
        processed_text = torch.tensor(stoi(text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))

    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)

    return (
        label_list,
        text_list,
        offsets,
        rule_assigned_instance_labels,
        rule_coverage_matrix,
        labelled_flag_matrix,
        rule_exemplar_matrix,
    )


# returing the created dataloaders to the caller function after making the vocab
def get_dataloaders(file_name, batch_size):
    global vocab, tokenizer

    print("Making the vocabulary...")

    data = pickle.load(open(f"data/{file_name}", "rb"))

    if vocab is None:
        data, vocab, tokenizer = make_vocab(data)

    print("Done!")

    data = random.sample(data, len(data))

    print("Creating dataloaders...")

    train_split = int(0.7 * len(data))
    valid_split = train_split + int(0.15 * len(data))
    train_data, valid_data, test_data = (
        data[:train_split],
        data[train_split:valid_split],
        data[valid_split:],
    )

    train_dataloader = DataLoader(
        train_data,  # type: ignore
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        # generator=torch.Generator(device=device),
    )

    valid_dataloader = DataLoader(
        valid_data,  # type: ignore
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        # generator=torch.Generator(device=device),
    )

    test_dataloader = DataLoader(
        test_data,  # type: ignore
        batch_size=batch_size,
        # shuffle=True,
        collate_fn=collate_batch,
        # generator=torch.Generator(device=device),
    )

    print("Done!")

    return train_dataloader, valid_dataloader, test_dataloader, vocab
