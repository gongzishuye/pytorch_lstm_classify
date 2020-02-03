# -*- coding: utf-8 -*-
"""
负责处理data相关，直接生成可供pytorch接口使用的数据
"""
import random
import torch

VOCAB_PATH = "data/vocab.txt"
TRAIN_DATA_PATH = "data/jd_comments_emotions.out"


def build_vocab():
  vocabs = open(VOCAB_PATH, encoding="utf-8").read().strip().split("\n")
  vocabs.insert(0, "UNK")
  vocabs.insert(0, "PAD")
  char2index = {char: index for index, char in enumerate(vocabs)}
  index2char = {index: char for index, char in enumerate(vocabs)}
  return char2index, index2char

class DataIterater(object):
  def __init__(self, batches, batch_size, device):
    self.batch_size = batch_size
    self.batches = batches
    self.n_batches = len(batches) // batch_size
    self.residue = False  # 记录batch数量是否为整数
    if len(batches) % self.n_batches != 0:
      self.residue = True
    self.index = 0
    self.device = device

  def __iter__(self):
    return self

  def __next__(self):
    if self.residue and self.index == self.n_batches:
      batches = self.batches[self.index * self.batch_size: len(self.batches)]
      self.index += 1
      batches = self._to_tensor(batches)
      return batches

    elif self.index >= self.n_batches:
      self.index = 0
      raise StopIteration
    else:
      batches = self.batches[
                self.index *
                self.batch_size: (self.index + 1) * self.batch_size]
      self.index += 1
      batches = self._to_tensor(batches)
      return batches

  def __len__(self):
    pass

  def _to_tensor(self, datas):
    x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
    y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
    return x, y


def create_iter(config, eval_prop=0.1):
  datas = open(TRAIN_DATA_PATH, encoding="utf-8")\
    .read().strip().split("\n")
  char2index, index2char = build_vocab()
  def sentence2index(sentence):
    sentence = [char for char in sentence if '\u4e00' <= char <= '\u9fff']
    padding_num = 0 if len(sentence) >= config.seq_lenght \
      else config.seq_lenght - len(sentence)
    sentence = sentence[:config.seq_lenght] + ["PAD"] * padding_num
    indexes = [char2index.get(char, char2index["UNK"])
               for char in sentence]
    return indexes
  label2index = {label: index
                 for index, label in enumerate(config.labels)}
  datas = [data.strip().split("\t") for data in datas]
  datas = [(sentence2index(data[0]), label2index[data[1]])
           for data in datas]
  random.seed(1)
  random.shuffle(datas)
  split_point = int(len(datas) * eval_prop)
  datas_train = datas[split_point:]
  datas_eval  = datas[:split_point]

  datas_train_iter = DataIterater(
    datas_train, config.batch_size, config.device
  )
  datas_eval_iter  = DataIterater(
    datas_eval, config.batch_size, config.device
  )
  return datas_train_iter, datas_eval_iter


if __name__ == "__main__":
  from model import Config
  config = Config(128)
  datas_train_iter, datas_eval_iter = create_iter(config)
  for data in datas_eval_iter:
    print(data[0])