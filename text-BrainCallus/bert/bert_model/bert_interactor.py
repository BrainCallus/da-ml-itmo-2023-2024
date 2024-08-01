import datetime

import pandas as pd
from torch import nn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import precision_recall_fscore_support
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
import torch
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import math
from functools import cmp_to_key

from bert.bert_model.bert_impl import BertImpl
from bert.bert_model.logger import Logger


class BertInteractor:
    def __init__(self, data, test_part, target_col, labels, bins):
        self.data = data
        self.test_part = test_part
        self.target_col = target_col
        self.labels = labels
        self.bins = bins
        self.bert = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.logger = Logger(BertInteractor.__name__)
        self.logger.info("Activated BertInteractor")

    def mark_dataset(self):
        analyzer = SentimentIntensityAnalyzer()
        self.data['sentiment'] = self.data[self.target_col].apply(lambda x: analyzer.polarity_scores(x)['compound'])
        self.data['label'] = pd.cut(self.data['sentiment'], bins=self.bins, labels=self.labels)
        self.logger.info(f'Marked data with labels={self.labels} bins={self.bins}')

    def init_bert(self):
        self.bert = BertImpl(self.data, self.labels.__len__(), self.test_part)

    def train_model(self, bert, optimizer, batch_size, epochs):
        self.logger.warn("Train process may took a time. Do not interrupt it")
        train_dataset = self.build_dataset(bert.train_data)
        bert.set_train_loader(train_dataset, batch_size)
        self.logger.info("Model training started")
        bert.train_model(optimizer, epochs)

    def test_model(self, bert):
        test_dataset = self.build_dataset(bert.test_data)
        bert.set_test_loader(test_dataset)
        all_labels, all_preds = bert.eval_model()
        return all_labels, all_preds

    def find_best_optimizer_and_train(self, epochs, test_slice=300, extended=False):
        lrs = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]  # , 0.01, 0.05, 0.1] - useless, long
        batches = self.get_batches(test_slice)
        self.logger.info(f'Searching for best lrs and batches for AdamW from lr = {lrs}, batch_sizes = {batches}.\n'
                         + 'Keep calm, pour yourself some coffe and patiently wait)))')
        results = sorted(self.test_optimizers(lrs, batches, epochs, test_slice, extended),
                         key=cmp_to_key(self.cmp_optim_res_by_f1 if extended else self.cmp_optim_res_by_loss))
        best = results[0]

        self.logger.info(f'Best lr: {best.lr}, best batch: {best.batch}, finished in {best.time}')
        self.init_bert()
        self.train_model(self.bert, best.optimizer, best.batch, epochs)
        return results

    def get_batches(self, test_slice):
        batches = [1, 3]
        i = 2
        while 2 ** (i + 1) < self.test_slice(test_slice) * 0.15:
            inc = math.floor(2 ** i / i)
            j = 2 ** i + inc
            while j < 2 ** (i + 1):
                batches.append(j)
                j += inc
            i += 1

        return batches

    def test_slice(self, test_slice):
        return min(self.data.__len__(), test_slice)

    def test_optimizers(self, lrs, batches, epochs, test_slice, extended):
        results = []
        for lr in lrs:  # ~ lrs.len * log n

            results, idx = self.ternary_search(-1, batches.__len__(), lambda x: batches[x], results, lr,
                                               epochs, test_slice, extended)  # ~ log(log(0.15* n))
            if not any(map(lambda x: (x.f1_criteria() if extended else x.loss_criteria()), results)):
                self.logger.warn('All batches produced not enough accurate results')

        return results

    def ternary_search(self, left, right, idx_fun, results, lr, epochs, test_slice, extended):
        while right - left > 1:
            i1 = math.floor((2 * left + right) / 3)
            i2 = math.floor((left + 2 * right) / 3)
            r1 = self.per_batch_test(idx_fun(i1), lr, epochs, test_slice, extended)
            r2 = self.per_batch_test(idx_fun(i2), lr, epochs, test_slice, extended)
            if self.cmp_optim_res_by_f1(r1, r2) if extended else self.cmp_optim_res_by_loss(r1, r2):
                right = i2
                self.logger.info(f'Better batch_size={idx_fun(i1)}')
            else:
                left = i1
                self.logger.info(f'Better batch_size={idx_fun(i2)}')
            results.append(r1)
            results.append(r2)
        return results, math.floor((left + right) / 2)

    def per_batch_test(self, batch, lr, epochs, test_slice, extended):
        bert = BertImpl(self.data[:self.test_slice(test_slice)], self.labels.__len__(), self.test_part)
        optimizer = torch.optim.AdamW(bert.model.parameters(), lr=lr)
        self.logger.info(f"Testing optimizer with lr={lr} batch={batch}")
        avg_loss = 1000
        start = datetime.datetime.now()
        if extended:
            self.extended_batch_test(bert, optimizer, batch, epochs)
        else:
            avg_loss = self.run_epoch(bert, optimizer, batch)
        total_time = datetime.datetime.now() - start
        f1 = self.get_weighted_f_avg(self.test_model(bert)) if extended else -1
        self.logger.info(
            f'Optimizer with lr={lr} and batch={batch} finished in time: {total_time} with ' + (
                f'f1: {f1}' if extended else f'avg_loss: {avg_loss} '))
        return self.OptimizerResult(optimizer, bert, f1, avg_loss, lr, batch, total_time)

    def extended_batch_test(self, bert, optimizer, batch, epochs):
        self.train_model(bert, optimizer, batch, epochs)

    def run_epoch(self, bert, optimizer, batch):
        train_dataset = self.build_dataset(bert.train_data)
        bert.set_train_loader(train_dataset, batch)
        total_loss = bert.train_iter(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), optimizer,
                                     nn.CrossEntropyLoss())
        return total_loss / len(bert.train_loader)

    def prepare_data(self):
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.data['prepared_' + self.target_col] = self.data[self.target_col].apply(lambda x: self.process_comment(x))

    def build_dataset(self, data, max_length=128):
        input_ids = []
        attention_masks = []
        for item in tqdm(data['prepared_' + self.target_col]):
            encoding = self.encode_item(item, max_length)
            input_ids.append(encoding['input_ids'])
            attention_masks.append(encoding['attention_mask'])

        return TensorDataset(self.cat(input_ids), self.cat(attention_masks), self.extract_labels(data))

    def process_comment(self, comment):
        return self.lemmatize(self.remove_stopwords(comment))

    def encode_item(self, item, max_length):
        return self.tokenizer.encode_plus(
            item,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

    @staticmethod
    def get_weighted_f_avg(labels_preds):
        labels, preds = labels_preds
        return precision_recall_fscore_support(
            labels,
            preds,
            labels=None,
            average='weighted',
            sample_weight=None, zero_division='warn'
        )[2]

    @staticmethod
    def cat(arr):
        return torch.cat(arr, dim=0)

    @staticmethod
    def extract_labels(data):
        return torch.tensor(data['label'].astype('category').cat.codes.values, dtype=torch.long)

    @staticmethod
    def remove_stopwords(text):
        stop_words = set(stopwords.words('russian')).union(stopwords.words('english'))
        tokens = text.split()
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        return ' '.join(filtered_tokens)

    @staticmethod
    def lemmatize(text):
        lemmatizer = WordNetLemmatizer()
        tokens = text.split()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(lemmatized_tokens)

    class OptimizerResult:
        def __init__(self, optimizer, pretrained_model, f1, avg_loss, lr, batch, time):
            self.optimizer = optimizer
            self.pretrained_model = pretrained_model
            self.f1 = f1
            self.avg_loss = avg_loss
            self.lr = lr
            self.batch = batch
            self.time = time

        def f1_criteria(self):
            return self.f1 >= 0.8

        def loss_criteria(self):
            return self.avg_loss < 0.55

        def f1_metric(self):
            return self.f1 / self.time.total_seconds() / 60

        def loss_metric(self):
            return self.time * self.avg_loss

    @staticmethod
    def cmp_optim_res_by_loss(first: OptimizerResult, second: OptimizerResult):
        return first.loss_criteria() if first.loss_criteria() ^ second.loss_criteria() else (
                first.loss_metric() < second.loss_metric())

    @staticmethod
    def cmp_optim_res_by_f1(first: OptimizerResult, second: OptimizerResult):
        return first.f1_criteria() if first.f1_criteria() ^ second.f1_criteria() else (
                first.f1_metric() > second.f1_metric())
