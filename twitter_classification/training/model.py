from torch.utils.data import DataLoader
import torch
from transformers import RobertaForSequenceClassification, AdamW
from tqdm import tqdm

class ModelTrainer():
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3, output_attentions=False, output_hidden_states=False)

        # Use GPU if possible
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

        # AdamW optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=1e-4)

    def train(self, epochs: int):
        self.model.train()
        for epoch in range(epochs):
            loop = tqdm(self.dataloader, leave=False)
            for batch in loop:
                # Initialize gradients
                self.optimizer.zero_grad()

                # Get tensor batches
                batch = tuple(b.to(self.device) for b in batch)
                inputs = {
                    'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2],
                }

                outputs = self.model(**inputs)

                # Extract and calculate loss
                loss = outputs.loss
                loss.backward()
                # Update parameters
                self.optimizer.step()

                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())

        self.model.save_pretrained('../trained_models/trained_RoBERTa_model')