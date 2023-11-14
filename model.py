from torch import nn


class ClassificationNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_nodes, num_class):
        super(ClassificationNetwork, self).__init__()

        # self.glove_embedding = nn.EmbeddingBag.from_pretrained(
        #     embedding_matrix, freeze=True
        # )
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.dense_block = nn.Sequential(
            nn.Linear(embed_dim, num_nodes),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(num_nodes, num_class),
        )

        self.embedding.apply(self.init_weights)
        self.dense_block.apply(self.init_weights)

    def init_weights(self, layer):
        initrange = 0.5

        if isinstance(layer, nn.Linear or nn.EmbeddingBag):
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.zero_()

    def forward(self, text, offsets):
        # combined_embedding = torch.cat(
        #     [self.embedding(text, offsets), self.glove_embedding(text, offsets)],
        #     dim=-1,
        # )
        embedded = self.embedding(text, offsets)
        return self.dense_block(embedded)


class RuleNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_nodes, num_class):
        super(RuleNetwork, self).__init__()

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, num_nodes[0]),
            nn.LayerNorm(num_nodes[0]),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(num_nodes[0], num_nodes[1]),
            nn.LayerNorm(num_nodes[1]),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(num_nodes[1], num_class),
        )

        self.embedding.apply(self.init_weights)
        self.MLP.apply(self.init_weights)

    def init_weights(self, layer):
        initrange = 0.5

        if isinstance(layer, nn.Linear or nn.EmbeddingBag):
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.MLP(embedded)
