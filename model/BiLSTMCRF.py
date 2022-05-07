#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: BiLSTMCRF.py
@time:2021/11/20
@description:
"""
from argparse import ArgumentParser
from typing import Union, Dict, List, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
import torch
from pytorch_lightning import LightningModule
from torch.optim import RAdam
from torch import nn, Tensor
from torchcrf import CRF
from torch.nn.utils import rnn as rnn_utils


class BiLSTMCRF(LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=5e-03)
        parser.add_argument('--hidden_dim', type=int, default=32)
        parser.add_argument('--data_path', type=str, default="data/corpus")
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--weight_decay", type=float, default=9e-3)
        parser.add_argument("--char_embedding_size", type=int, default=60)
        parser.add_argument("--experiment", type=bool, default=False)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.hyper_params = hparams
        self.lr = hparams.lr
        self.hidden_size = self.hyper_params.hidden_dim // 2
        self.num_layers = 1
        self.bi_directions = True
        self.num_directions = 2 if self.bi_directions else 1
        self.word_emb = nn.Embedding(self.hyper_params.vocab_size, self.hyper_params.char_embedding_size)

        self.lstm = nn.LSTM(input_size=self.hyper_params.char_embedding_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=self.bi_directions)
        self.dense = nn.Linear(self.hyper_params.hidden_dim,
                               self.hyper_params.tag_size)
        self.id2char = self.hyper_params.id2char
        self.idx2tag = self.hyper_params.idx2tag
        self.crf = CRF(num_tags=self.hyper_params.tag_size, batch_first=True)
        self.dropout = nn.Dropout(self.hyper_params.dropout)
        self.hidden_state = self._init_hidden(self.hyper_params.batch_size)

    def configure_optimizers(self):
        """
        配置优化器
        :return:
        """
        optimizer = RAdam(self.parameters(),
                          lr=self.lr,
                          weight_decay=self.hyper_params.weight_decay)
        return optimizer

    def forward_train(self, sentences_idx, tags_idx):
        """
        model train
        :param sentences_idx:
        :param tags_idx:
        :return:
        """
        feats = self._get_lstm_features(sentences_idx)
        mask = tags_idx != 1
        loss = self.crf(feats, tags_idx, mask=mask, reduction='mean')
        return -loss

    def _get_lstm_features(self, sentences_idx):
        char_inputs = self.word_emb(sentences_idx)
        inputs = self.dropout(char_inputs)
        sentences_len = torch.empty(sentences_idx.shape[0], dtype=torch.int64)
        for idx, elem in enumerate(sentences_idx):
            sentences_len[idx] = sentences_idx.size()[1] - (elem == 1).sum(dim=-1).item()
        embed_input_packed = rnn_utils.pack_padded_sequence(inputs, sentences_len, batch_first=True)
        output, _ = self.lstm(embed_input_packed)
        output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True)
        feats = self.dense(output)
        return feats

    def _init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers * self.num_layers, batch_size, self.hidden_size, device=self.device),
                torch.zeros(self.num_layers * self.num_layers, batch_size, self.hidden_size, device=self.device))

    def _get_batch_info(self, batch):
        this_batch_size = batch.word[0].size()[0]
        sentences_idx = batch.word[0].view(this_batch_size, -1)
        tags = batch.tag[0].view(this_batch_size, -1)
        sentences_length = batch.word[1]
        return sentences_idx, tags, sentences_length

    def forward(self, sentences_idx):
        """
        模型落地推理
        :param sentences_idx:
        :return:
        """
        return self._decode(sentences_idx)

    def _decode(self, sentences_idx):
        """
        模型实际预测函数
        :param sentences_idx:
        :return:
        """
        feats = self._get_lstm_features(sentences_idx)
        results = self.crf.decode(feats)
        result_tensor = []
        for result in results:
            result_tensor.append(torch.tensor(result))
        return torch.stack(result_tensor)

    def training_step(self, batch, batch_idx, optimizer_idx=None) -> Union[int,
                                                                           Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        """
        模型训练的前向传播过程
        :param batch:批次数据
        :param batch_idx:
        :param optimizer_idx:
        :return:
        """
        sentences_idx, tags, sentences_length = batch
        loss = self.forward_train(sentences_idx, tags)
        res = {"log": {"loss": loss}, "loss": loss}
        return res

    def validation_step(self, batch, batch_idx) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        """
        开发集数据验证过程
        :param batch: 批次数据
        :param batch_idx:
        :return:
        """
        sentences_idx, tags, sentences_lengths = batch
        loss = self.forward_train(sentences_idx, tags)
        loss = loss.mean()
        return {"sentence_lengths": sentences_lengths, 'sentence': sentences_idx, "target": tags,
                "pred": self._decode(sentences_idx), "loss": loss}

    def validation_epoch_end(self, outputs: Union[List[Dict[str, Tensor]],
                                                  List[List[Dict[str, Tensor]]]]) -> Dict[str, Dict[str, Tensor]]:
        """
        验证数据集
        :param outputs: 所有batch预测结果 validation_step的返回值构成的一个list
        :return:
        """
        return self._decode_epoch_end(outputs)

    def _decode_epoch_end(self, outputs: Union[List[Dict[str, Tensor]],
                                               List[List[Dict[str, Tensor]]]]) -> Dict[str, Dict[str, Tensor]]:
        """
        对批次预测的结果进行整理，评估对应的结果
        :return:
        """
        ner_results = []
        gold_list, pred_list = [], []  # 原始标签以及模型预测结果
        for batch_result in outputs:
            batch_size = batch_result['sentence_lengths'].shape[0]
            for i in range(batch_size):
                res = []  # char gold pred
                sentence_gold, sentence_pred = [], []
                for j in range(batch_result['sentence_lengths'][i].item()):
                    char = self.id2char[batch_result['sentence'][i][j]]
                    gold = self.idx2tag.get(batch_result['target'][i][j].item())
                    pred = self.idx2tag.get(batch_result['pred'][i][j].item())
                    if gold == "<pad>":
                        break
                    res.append(" ".join([char, gold, pred]))
                    sentence_gold.append(gold)
                    sentence_pred.append(pred)
                ner_results.append(res)
                gold_list.append(sentence_gold)
                pred_list.append(sentence_pred)
        print("\n", classification_report(gold_list, pred_list))
        f1 = torch.tensor(f1_score(gold_list, pred_list))
        tqdm_dict = {'val_f1': f1}
        results = {"progress_bar": tqdm_dict, "log": {'val_f1': f1, "step": self.current_epoch}}
        self.log("val_f1", f1)
        return results

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        """
        程序测试模块
        :param batch:
        :param batch_idx:
        :return:
        """
        sentences_idx, tags, sentences_lengths = batch
        loss = self.forward_train(sentences_idx, tags)
        loss = loss.mean()
        return {"sentence_lengths": sentences_lengths,
                'sentence': sentences_idx, "target": tags,
                "pred": self._decode(sentences_idx), "loss": loss}

    def test_epoch_end(self, outputs: Union[List[Dict[str, Tensor]],
                                            List[List[Dict[str, Tensor]]]]) -> Dict[str, Dict[str, Tensor]]:
        """
        测试集的评估
        :param outputs:测试集batch预测完成结果
        :return:
        """
        return self._decode_epoch_end(outputs)
