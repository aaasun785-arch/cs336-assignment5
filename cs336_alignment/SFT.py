import torch
import json
from torch.utils.data import Dataset,DataLoader
import random

from xopen import xopen

class SFT_dataset(Dataset):
    def __init__(self,tokenizer,dataset_path,seq_length,shuffle):
        super().__init__()
        self.seq_length=seq_length
        with open("cs336_alignment/prompts_safety/alpaca_sft.prompt","r",encoding="utf-8") as f:
            template=f.read()
        documents=[]
        with xopen(dataset_path,"rt")as f:
            for line in f:
                example=json.loads(line)
                text=template.format(instruction=example["prompt"].strip(),
                                     response=example["response"].strip()).strip()
                documents.append(text)

        if shuffle:
            random.shuffle(documents)
        all_tokens=[]
        
        for document in documents:
            token_ids = tokenizer.encode(document)

            all_tokens.extend(token_ids)
            all_tokens.append(tokenizer.eos_token_id)
            
        self.tokens=torch.tensor(all_tokens,dtype=torch.long)
    def __len__(self):
        return (len(self.tokens)-1)//self.seq_length
    def __getitem__(self, i):
        start=i*self.seq_length
        end=start+self.seq_length
        input_ids=self.tokens[start:end]
        labels=self.tokens[start+1:end+1]
        return {"input_ids":input_ids,
                "labels":labels}

def iterate_batches(dataset,
                    batch_size,
                    shuffle):
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)