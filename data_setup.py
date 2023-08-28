import torch
import random
import pickle
import re
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader

# import torchtext.vocab as vocab

RANDOM_STATE = 695195
device = "cuda:0"  # if torch.cuda.is_available() else "cpu"
# device = "cpu"

random.seed(RANDOM_STATE)
torch.set_default_device(device)

vocab = None
tokenizer = None


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


def yield_tokens(data_iter, tokenizer):
    for text, _ in data_iter:
        yield tokenizer(text)


def make_vocab(data):
    tokenizer = get_tokenizer("spacy")

    data = [[preprocess(i[0]), i[1]] for i in data]

    vocab = build_vocab_from_iterator(yield_tokens(data, tokenizer), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    return vocab, tokenizer


def stoi(x):
    global vocab, tokenizer
    return vocab(tokenizer(x))  # type: ignore


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


def get_dataloaders(file_name, batch_size):
    global vocab, tokenizer

    data = pickle.load(open(f"data/{file_name}", "rb"))

    if vocab is None:
        vocab, tokenizer = make_vocab(data)

    data = random.sample(data, len(data))

    split = 0.3
    split_index = int(len(data) * (1 - split))
    train_data, test_data = data[:split_index], data[split_index:]

    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch, generator=torch.Generator(device=device)  # type: ignore
    )

    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch, generator=torch.Generator(device=device)  # type: ignore
    )

    return train_dataloader, test_dataloader, len(vocab)
