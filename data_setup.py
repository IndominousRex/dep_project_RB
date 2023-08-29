import torch
import random
import pickle
import re
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader

RANDOM_STATE = 12345
device = "cuda:0"  # if torch.cuda.is_available() else "cpu"

random.seed(RANDOM_STATE)
torch.set_default_device(device)

vocab = None
tokenizer = None


# function to preprocess the text by removing links, numbers, hashtags, symbols and emoticons if any
def preprocess(text):
    text = re.sub(r"@[A-Za-z0-9]+", "", text)
    text = re.sub(r"[0-9]+", "", text)
    text = re.sub(r"#[A-Za-z0-9]*", "", text)
    text = re.sub(r"https?:\/\/\S+", "", text)
    text = re.sub(r"[\n]", " ", text)
    text = re.sub(r"['\",*%$#()]", " ", text)
    text = re.sub(r"[.]", " . ", text)
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

    return vocab, tokenizer


# converting a token or a list of tokens into numbers
def stoi(x):
    global vocab, tokenizer
    return vocab(tokenizer(x))  # type: ignore


# function to create the input tensor and the offsets from a input batch
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]

    for _text, _label in batch:
        label_list.append(_label)
        processed_text = torch.tensor(stoi(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))

    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)

    return label_list.to(device), text_list.to(device), offsets.to(device)


# returing the created dataloaders to the caller function after making the vocab
def get_dataloaders(file_name, batch_size):
    global vocab, tokenizer

    print("Making the vocabulary...")

    data = pickle.load(open(f"data/{file_name}", "rb"))

    if vocab is None:
        vocab, tokenizer = make_vocab(data)

    print("Done!")

    data = random.sample(data, len(data))

    print("Creating dataloaders...")

    split = 0.3
    split_index = int(len(data) * (1 - split))
    train_data, test_data = data[:split_index], data[split_index:]

    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch, generator=torch.Generator(device=device)  # type: ignore
    )

    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch, generator=torch.Generator(device=device)  # type: ignore
    )

    print("Done!")

    return train_dataloader, test_dataloader, len(vocab)
