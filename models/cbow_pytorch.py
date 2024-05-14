import torch
import torch.nn as nn
import torch.optim as optim
from random import shuffle
from tqdm import tqdm

class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size+1, embedding_dim, padding_idx=0, max_norm=1)
        self.linear1 = nn.Linear(embedding_dim, vocab_size+1)
        self.act1 = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        out = self.embeddings(inputs)
        out = torch.mean(out, dim=1)
        out = self.linear1(out)
        out = self.act1(out)
        return out

class CbowModel:

    def __init__(self, max_vocab_size, device, embedding_dim=100, batch_size=128, window=10, epochs=1):
        self.max_vocab_size = max_vocab_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.window = window
        self.data = []
        self.word_to_ix = {'<PAD>': 0}
        self.ix_to_word = {0: '<PAD>'}
        self.epochs = epochs

        self.model = CBOW(self.max_vocab_size, self.embedding_dim)
        # self.loss_function = nn.NLLLoss()
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        # device:
        self.device = device
        self.model.to(self.device)

        print(f"Device using: {self.device}")
        
    def make_context_vector(self, context):
        idxs = [self.word_to_ix[w] for w in context if w in self.word_to_ix]
        while (len(idxs) < self.window * 2):
            idxs.append(0)
        return torch.tensor(idxs, dtype=torch.long)

    def train(self, context, anchor):

        # Re-initialize data:
        self.data = []

        for i in range(len(context)):
            if anchor[i] not in self.word_to_ix:
                self.word_to_ix[anchor[i]] = len(self.word_to_ix)
                self.ix_to_word[self.word_to_ix[anchor[i]]] = anchor[i]

            self.data.append((self.make_context_vector(context[i]),
                              torch.tensor(self.word_to_ix[anchor[i]], dtype=torch.long)))

        self.model.train()

        for epoch in range(self.epochs):  # You can choose a different number of epochs
            total_loss = 0

            shuffle(self.data)
            # batches = [self.data[i:i+self.batch_size] for i in range(0, min(1024, len(self.data)), self.batch_size)]
            batches = [self.data[i:i+self.batch_size] for i in range(0, len(self.data), self.batch_size)]

            for i in tqdm(range(len(batches))):
                context_batch, target_batch = zip(*batches[i])
                context_idxs = context_batch
                context_idxs_tensor = torch.stack(context_idxs).to(self.device)  # Shape: [batch_size, context_size]

                target_idxs = target_batch
                target_tensor = torch.stack(target_idxs).to(self.device)
                # target_tensor = torch.tensor(target_idxs, dtype=torch.long, device="mps")

                self.model.zero_grad()
                log_probs = self.model(context_idxs_tensor)
                loss = self.loss_function(log_probs, target_tensor)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            # print(f"Epoch {epoch}: Total Loss: {total_loss}")

    def predict(self, contexts):
        if len(self.word_to_ix) == 1:
            return {}

        self.model.eval()

        #TESTING
        context_vectors = torch.stack([self.make_context_vector(context) for context in contexts]).to(self.device)
        preds = self.model(context_vector).detach().exp()[0]
        s = sum([p.item() for i, p in enumerate(preds) if i in self.ix_to_word])
        return {self.ix_to_word[i]: p.item()/s for i, p in enumerate(preds) if i in self.ix_to_word}

if __name__ == "__main__":

    CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
    raw_text = """We are about to study the idea of a computational process.
    Computational processes are abstract beings that inhabit computers.
    As they evolve, processes manipulate other abstract things called data.
    The evolution of a process is directed by a pattern of rules
    called a program. People create programs to direct processes. In effect,
    we conjure the spirits of the computer with our spells.""".lower().split()

    # By deriving a set from `raw_text`, we deduplicate the array
    vocab = set(raw_text)
    vocab_size = len(vocab)

    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {ix:word for ix, word in enumerate(vocab)}

    data = []
    for i in range(CONTEXT_SIZE, len(raw_text) - CONTEXT_SIZE):
        context = (
            [raw_text[i - j - 1] for j in range(CONTEXT_SIZE)]
            + [raw_text[i + j + 1] for j in range(CONTEXT_SIZE)]
        )
        target = raw_text[i]
        data.append((context, target))
    print(data[:5])

    model = CbowModel(100, window=CONTEXT_SIZE)
    first = False
    probs = []
    for context, anchor in data:
        if anchor not in model.vocab:
            p = 0
        else:
            # print(model.predict(context))
            p = model.predict(context)[anchor]
        probs.append(p)
        print("prob = {}".format(p))
        model.train(context, anchor)

    print('AVG prob', sum(probs)/len(probs))

    # # Set hyperparameters
    # embedding_dim = 100  # You can choose a different value

    # # Create your model
    # model = CBOW(vocab_size, embedding_dim)
    # loss_function = nn.NLLLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001)

    # # Training loop
    # for epoch in range(5000):  # You can choose a different number of epochs
    #     total_loss = 0
    #     for context, target in data:
    #         context_idxs = make_context_vector(context, word_to_ix)
    #         
    #         model.zero_grad()
    #         log_probs = model(context_idxs)
    #         loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
    #         loss.backward()
    #         optimizer.step()

    #         total_loss += loss.item()
    #     print(f"Epoch {epoch}: Total Loss: {total_loss}")

    # #TESTING
    # context = ['people','create','to', 'direct']
    # context_vector = make_context_vector(context, word_to_ix)
    # a = model(context_vector).detach()

    # #Print result
    # print(f'Raw text: {" ".join(raw_text)}\n')
    # print(f'Context: {context}\n')
    # print(f'Prediction: {ix_to_word[torch.argmax(a[0]).item()]}')
