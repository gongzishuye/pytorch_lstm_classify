# -*- coding: utf-8 -*-
"""
创建model
"""
import torch
from torch import nn
import os

class Config(object):
  def __init__(self, vocab_size):
    self.device = torch.device("cpu")
    self.labels = ["POS", "NEG"]
    # 训练参数
    self.num_layers = 3
    self.batch_size = 32
    self.lr = 0.001
    self.save_path = "result"
    self.init()

    # 模型参数
    self.vocab_size = vocab_size
    self.hidden_size = 128
    self.embedding_size = 128
    self.num_classes = 2
    self.seq_lenght = 64

  def init(self):
    if not os.path.exists(self.save_path):
      os.mkdir(self.save_path)


class TextRnn(nn.Module):

  def __init__(self, config):
    super(TextRnn, self).__init__()
    self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
    self.lstm      = nn.LSTM(
      input_size = config.embedding_size,
      hidden_size = config.hidden_size,
      num_layers = config.num_layers,
      bias = True,
      batch_first = True,
      dropout = 1.0,
      bidirectional = True
    )
    self.linear    = nn.Linear(
      in_features = config.hidden_size * 2,
      out_features = config.num_classes
    )
    self.softmax   = nn.Softmax()

  def forward(self, input_data):
    """
    :param input_data: [batch_size, seq_length]
    :return:
    """
    # [batch_size, seq_length, embedding_size]
    output = self.embedding(input_data)
    # output [batch_size, seq_length, 2*hidden_size]
    output, _ = self.lstm(output)
    # [batch_size, 2*hidden_size]
    output = output[:, -1, :].squeeze(dim=1)
    # [batch_size, num_classes]
    output = self.linear(output)
    output = self.softmax(output)
    return output


if __name__ == "__main__":
  config = Config(1000)
  textRnn = TextRnn(config)
  input_data = torch.tensor([[1,2,3,4]])
  output_data = textRnn(input_data)
  print(output_data.size())