from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torch.nn as nn

from bert.bert_model.logger import Logger


class BertImpl:

    def __init__(self, data, num_labels, test_part=0.3):
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=num_labels)
        self.train_data, self.test_data = train_test_split(data, test_size=test_part, random_state=42)
        self.logger = Logger(BertImpl.__name__)
        self.train_loader = None
        self.test_loader = None
        self.logger.info("BertImpl initialized")

    def set_train_loader(self, train_dataset, batch_size):
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    def set_test_loader(self, train_dataset, batch_size=-1):
        if self.train_loader is None and batch_size == -1:
            self.logger.warn('No train loader found nor correct batch_size. Set default batch_size = 8')
            batch_size = 8
        self.test_loader = DataLoader(train_dataset,
                                      batch_size=self.train_loader.batch_size if batch_size == -1 else batch_size)

    def train_model(self, optimizer, num_epochs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            total_loss = self.train_iter(self.device, optimizer, criterion)
            self.logger.info(
                f'Epoch {epoch + 1}/{num_epochs}, '
                + 'Total loss: {total_loss:.4f}, '.format(total_loss=total_loss)
                + 'Avg. loss: {avg_loss: .4f}'.format(avg_loss=total_loss / len(self.train_loader))
            )
        self.logger.info("Finished training")

    def eval_model(self):
        if self.device is None:
            raise ValueError('Model was not trained')
        if self.test_loader is None:
            raise ValueError('No test loader found. Set test_loader first')

        self.model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                labels, preds = self.eval_iter(batch, self.device)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        return all_labels, all_preds

    def eval_iter(self, batch, device):
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = self.model(input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)
        return labels, preds

    def train_iter(self, device, optimizer, criterion):
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.train_loader):
            total_loss += self.batch_iter(batch, device, optimizer, criterion)

        return total_loss

    def batch_iter(self, batch, device, optimizer, criterion):
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = self.model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        return loss.item()
