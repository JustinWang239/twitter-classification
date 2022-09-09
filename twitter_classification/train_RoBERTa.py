from dataset_extract_transform.preprocess import Loader
from training.model import ModelTrainer
from training.tokenizer import Tokenizer

class Trainer():
    def __init__(self, path: str = './offensive_tweet_dataset/labeled_data.csv'):
        self.loader = Loader(path)

    def train(self, batch_size: int = 32, epochs: int = 10):
        self.loader.load()
        self.loader.preprocess()
        train_set, train_labels = self.loader.get_training_data()
        tokenizer = Tokenizer(train_set, train_labels)

        dataloader = tokenizer.data_loader(batch_size)

        model = ModelTrainer(dataloader)
        model.train(epochs)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
