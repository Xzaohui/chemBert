# import torch
# print(torch.__version__)
# print(torch.cuda.is_available())

# from gensim.models import word2vec

# sentences = word2vec.Text8Corpus('./data/smiles.txt')
# model = word2vec.Word2Vec(sentences,vector_size=256)
# model.save('./model/chemical256.w2v')

import torch
from model import lstm
import lstm_data

model=lstm()

print(model(lstm_data.data_frame['NR-AR']['train'][0][0].unsqueeze(0)))

