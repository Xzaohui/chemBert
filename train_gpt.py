from transformers import AutoModelForCausalLM,AutoTokenizer
import torch
import data_loader
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

device='cpu'
batch_size=1

tokenizer = AutoTokenizer.from_pretrained("msb-roshan/molgpt")
model = AutoModelForCausalLM.from_pretrained("msb-roshan/molgpt")
model.to(device=device)

data_frame=data_loader.train_data_gpt('smiles_product.json', tokenizer)
train_dataloader=DataLoader(data_frame['train'],batch_size=batch_size,shuffle=True ,num_workers = 0)

def train(epoch=1):
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_dataloader) * epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.06*total_steps,num_training_steps=total_steps)
    for _ in range(epoch):
        for i,(train_data,attention_mask,train_lab) in enumerate(train_dataloader):
            optimizer.zero_grad()
            model.train()
            loss=model(input_ids=train_data.to(device),labels=train_lab.to(device)).loss
            print(loss)
            loss.backward()
            optimizer.step()
            scheduler.step()
    model.save_pretrained('./models/ftgpt')
    tokenizer.save_pretrained('./models/ftgpt')

train(1)
