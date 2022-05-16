import torch
import pandas as pd
from collections import Counter


def load_words():
    train_df = pd.read_csv('niggers.txt')
    text = train_df['Joke'].str.cat(sep=' ')
    return text.split(' ')


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            args,
    ):
        self.args = args
        self.words = load_words()
        self.uniq_words = self.get_uniq_words()

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - self.args.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index + self.args.sequence_length]),
            torch.tensor(self.words_indexes[index + 1:index + self.args.sequence_length + 1]),
        )
