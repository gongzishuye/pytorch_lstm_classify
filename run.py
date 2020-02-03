# -*- coding: utf-8 -*-
"""
入口函数，进行train，test和predict
主要负责：
  1. 参数的组装
  2. 数据的处理
  3. train等入口函数的处理
"""
import argparse
from data_helper import create_iter, build_vocab
from train_eval import train
from model import Config, TextRnn
from predict import predict

parser = argparse.ArgumentParser(description='Lstm Text Classification')
parser.add_argument('--operation', default="train", type=str, help='train test or predict')
args = parser.parse_args()

char2index, index2char = build_vocab()
config = Config(len(char2index))
train_iter, eval_iter = create_iter(config)
model = TextRnn(config)

operation = args.operation
if operation == "train":
  print("*****[ begin to train ]*****")
  train(config, model, train_iter, eval_iter)
elif operation == "predict":
  print("*****[ begin to predict ]*****")
  predict(config, model)