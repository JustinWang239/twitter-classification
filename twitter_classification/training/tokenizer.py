import numpy as np
from transformers import RobertaTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

class Tokenizer():
    def __init__(self, values: np.ndarray, labels: np.ndarray):
        self.values = values
        self.labels = labels
        self.tokenize()

    def tokenize(self) -> None:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        self.tokens = tokenizer.batch_encode_plus(
            self.values,
            return_attention_mask=True,
            truncation=True,
            max_length=256,
            padding='max_length',
            return_tensors='pt'
        )

    def data_loader(self, batch_size: int) -> DataLoader:
        dataset = TensorDataset(
            torch.tensor(self.tokens['input_ids']),
            torch.tensor(self.tokens['attention_mask']),
            torch.tensor(self.labels)
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=RandomSampler(dataset)
        )

        return dataloader