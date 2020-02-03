# -*- coding: utf-8 -*-
"""
训练模型
"""
import torch
from sklearn import metrics
import os

def train(config, model, train_iter, eval_iter):
  """
  负责训练模型，并进行eval
  :param config: 配置参数
  :param model: 模型
  :param train_iter: 训练数据迭代器
  :param eval_iter: 验证数据迭代器
  :return:
  """
  model.train()
  cross_entropy = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

  for index, train_data in enumerate(train_iter):
    input_data, label = train_data
    model.zero_grad()

    output = model(input_data)
    # output.size(batch_size, num_classes) label.size(N)
    loss = cross_entropy(output, label)
    loss.backward()
    optimizer.step()

    if index % 100 == 0:
      ground_truth = label.data.cpu()
      predict = torch.argmax(output, dim=1)
      train_acc = metrics.accuracy_score(ground_truth, predict)
      print("train acc: {train_acc}; train loss: {loss}"
            .format(train_acc=train_acc, loss=loss.item()))
      torch.save(model.state_dict(), os.path.join(config.save_path, "model.ckpt"))
