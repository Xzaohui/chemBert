import torch

from model import BertModel
from transformers import BertConfig

config=BertConfig(vocab_size=9999)

model=BertModel(config)

print(model)


