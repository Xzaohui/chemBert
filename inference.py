from transformers import AutoTokenizer
import torch

device='gpu'

model_name='A-SUB'
model_name='COF'
model_name='D-SUB'

tokenizer = AutoTokenizer.from_pretrained("./model/bertMolE")
model=torch.load("./model/models/model_{}.pt".format(model_name))
model.to(device=device)
model.eval()


def inference(chem):
    inputs = tokenizer(chem, return_tensors="pt",padding="max_length", max_length=512, truncation=True)
    output=model(inputs['input_ids'].to(device),inputs['attention_mask'].to(device))
    return output[0].item()

if __name__ == '__main__':
    chem='C1=CC=C(C=C1)C2=CC=CC=C2'
    inference(chem)