import torch
# import numpy as np
# from model import BertModel
# from transformers import BertConfig


# config=BertConfig(vocab_size=9999)

# model=BertModel(config)

# print(model)

# tf=torch.randint(0,9999,(1,512))
# tf1=torch.randint(0,1,(1,512))
# print(tf)
# print(model(tf,tf1))

# 掩码这里使用了加号，对于不掩码的位置来说，掩码值为0
# 对于掩码的位置来说，掩码值为-10000。使用softmax层之后，可以让-10000处的值为0。

# print(torch.cuda.is_available())




