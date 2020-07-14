import os
import random
import torch
import torch.nn as nn
import random
import numpy as np

from argparse import ArgumentParser
from transformers import AdamW

from src.parameters import DEVICE, SAVED_CHECKPOINTS

from src.model import ColBERT
from src.utils import print_message, save_checkpoint


class TrainReader:
    def __init__(self, data_file):
        print_message("#> Training with the triples in", data_file, "...\n\n")
        self.reader = open(data_file, mode='r', encoding="utf-8")
        self.data = self.reader.readlines()
        self.data_size = len(self.data)
        self.indexs = np.arange(self.data_size, dtype=int).tolist()
        self.reader.close()
        random.shuffle(self.indexs)

    def get_minibatch(self, batch_idx, bsize):
        ret_lst = []
        start_pos = batch_idx * bsize
        if start_pos >= self.data_size:
            random.shuffle(self.indexs)
        for idx in range(bsize):
            pos = (start_pos + idx) % self.data_size
            d_idx = self.indexs[pos]
            line = self.data[d_idx]
            x = line.strip().split('\t')
            assert (len(x) == 3)
            ret_lst.append(x)
        return ret_lst


def manage_checkpoints(colbert, optimizer, batch_idx):
    if batch_idx % 200 == 0:
        save_checkpoint("colbert-", 0, batch_idx, colbert, optimizer)

    if batch_idx in SAVED_CHECKPOINTS:
        save_checkpoint("colbert-" + str(batch_idx) + ".dnn", 0, batch_idx, colbert, optimizer)


def train(args):
    colbert = ColBERT.from_pretrained('bert-base-uncased',
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity)
    colbert = colbert.to(DEVICE)
    colbert.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(colbert.parameters(), lr=args.lr, eps=1e-8)

    optimizer.zero_grad()
    labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE)

    reader = TrainReader(args.triples)
    train_loss = 0.0
    tmp_delat = 0.0
    for batch_idx in range(args.maxsteps):
        Batch = reader.get_minibatch(batch_idx, args.bsize)
        Batch = sorted(Batch, key=lambda x: max(len(x[1]), len(x[2])))

        for B_idx in range(args.accumsteps):
            size = args.bsize // args.accumsteps
            B = Batch[B_idx * size: (B_idx+1) * size]
            Q, D1, D2 = zip(*B)

            colbert_out = colbert(Q + Q, D1 + D2)
            colbert_out1, colbert_out2 = colbert_out[:len(Q)], colbert_out[len(Q):]

            out = torch.stack((colbert_out1, colbert_out2), dim=-1)

            positive_score, negative_score = round(colbert_out1.mean().item(), 2), round(colbert_out2.mean().item(), 2)
            tmp_delat += positive_score - negative_score
            # print("#>>>   ", positive_score, negative_score, '\t\t|\t\t', positive_score - negative_score)

            loss = criterion(out, labels[:out.size(0)])
            loss = loss / args.accumsteps
            loss.backward()

            train_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0)

        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 200 == 1:
            print_message(batch_idx, train_loss / (batch_idx+1), tmp_delat / 200)
            tmp_delat = 0
            save_checkpoint(os.path.join(args.output_dir, "colbert-" + str(batch_idx) + ".dnn"),
                            0, batch_idx, colbert, optimizer)
            # manage_checkpoints(colbert, optimizer, batch_idx+1)
