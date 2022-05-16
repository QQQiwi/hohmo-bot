import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(dataset, model, args, save_model=True, load_model=False, epoch_to_load=0):
    print("cuda: ", torch.cuda.is_available())
    model.to(DEVICE)
    model.train()

    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if load_model:
        checkpoint = torch.load(f"hohma_{epoch_to_load}.pth.tar")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    for epoch in range(args.max_epochs + 1):
        state_h, state_c = model.init_state(args.sequence_length)
        state_h = state_h.to(DEVICE)
        state_c = state_c.to(DEVICE)

        if save_model and (epoch % 5 == 0):
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, f"hohma_{epoch}.pth.tar")

        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 and batch == 0:
                print({'epoch': epoch, 'loss': loss.item()})


def predict(dataset, model, text, next_words=100, load_model=False, epoch_to_load=0):
    if load_model:
        checkpoint = torch.load(f"hohma_{epoch_to_load}.pth.tar")
        model.load_state_dict(checkpoint["state_dict"])

    model.to(DEVICE)
    model.eval()

    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))
    state_h = state_h.to(DEVICE)
    state_c = state_c.to(DEVICE)

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]]).to(DEVICE)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    words = ' '.join([str(elem) for elem in words])
    return words


parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--sequence-length', type=int, default=4)
args = parser.parse_args()

dataset = Dataset(args)
model = Model(dataset)

train(dataset, model, args)
print(predict(dataset, model, text='', load_model=True, epoch_to_load=10))
