# -*- coding: utf-8 -*-
import torch
import os
from data_helper import build_vocab

def predict(config, model):
  model_path = os.path.join(config.save_path, "model.ckpt")
  model.load_state_dict(torch.load(model_path))
  model.eval()
  char2index, _ = build_vocab()
  index2label = {index: label for index, label in enumerate(config.labels)}

  with torch.no_grad():
    while True:
      input_line = input("input your sentence >>>")
      input_index = [char2index[char] for char in input_line]
      input_index = torch.tensor([input_index]).to(config.device)
      output = model(input_index)
      output_arg = torch.argmax(output, dim=1)
      output_arg = output_arg.numpy().tolist()
      output = output.numpy().tolist()

      print("target label: [{label}] with props: {props}"
            .format(label=index2label[output_arg[0]], props=output[0]))