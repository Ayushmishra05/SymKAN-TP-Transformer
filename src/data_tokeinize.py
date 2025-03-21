from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast
import os
import torch
from transformers import RobertaTokenizerFast
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle

os.environ['USE_TF'] = "0"
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2),
        ("[PAD]", 0),
        ("[MASK]", 3)
    ],
)

trainer = trainers.BpeTrainer(
    vocab_size=5000,
    min_frequency=2,
    special_tokens=["[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

files = ["QED_data/QED_data.txt"]
tokenizer.train(files, trainer)
tokenizer.save("src/tokenizer/QED_tokenizer.json")
hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file="src/tokenizer/QED_tokenizer.json")
hf_tokenizer.cls_token = "[CLS]"
hf_tokenizer.sep_token = "[SEP]"
hf_tokenizer.pad_token = "[PAD]"
hf_tokenizer.mask_token = "[MASK]"
hf_tokenizer.save_pretrained("src/tokenizer/QED_tokenizer")
tokenizer_path = 'src/tokenizer/QED_tokenizer'
tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)



sample = "e_[ID](X)^(*) e_[ID](X)^(*) to e_[ID](X) e_[ID](X)"
tokens = tokenizer(sample)
print(tokens['input_ids'])

tokenizer.decode(tokens['input_ids'])

csv_path = 'QED_data/processed_dataset.csv'
df = pd.read_csv(csv_path)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

MAX_LENGTH = 44

def tokenize_function(example):
    input_tokens = tokenizer(
        example['text'],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="np"  
    )

    label_tokens = tokenizer(
        example['label'],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="np"
    )

    return {
        'input_ids': input_tokens['input_ids'][0],
        'attention_mask': input_tokens['attention_mask'][0],
        'labels': label_tokens['input_ids'][0]
    }


train_data = train_df.apply(tokenize_function, axis=1).tolist()
val_data = val_df.apply(tokenize_function, axis=1).tolist()
test_data = test_df.apply(tokenize_function, axis=1).tolist()


def convert_to_dict(data):
    return {
        'input_ids': np.stack([x['input_ids'] for x in data]),
        'attention_mask': np.stack([x['attention_mask'] for x in data]),
        'labels': np.stack([x['labels'] for x in data])
    }

train_dict = convert_to_dict(train_data)
val_dict = convert_to_dict(val_data)
test_dict = convert_to_dict(test_data)

dataset = DatasetDict({
    "train": Dataset.from_dict(train_dict),
    "validation": Dataset.from_dict(val_dict),
    "test": Dataset.from_dict(test_dict)
})

dataset.set_format(type='torch')

print(dataset['train'][0])


batch_size = 16

train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset['validation'], batch_size=batch_size)
test_loader = DataLoader(dataset['test'], batch_size=batch_size)

with open(r'src/Dataloaders/train_loader.pkl', 'wb') as f:
    pickle.dump(train_loader, f)

with open(r'src/Dataloaders/test_loader.pkl', 'wb') as f:
    pickle.dump(test_loader, f)

with open(r'src/Dataloaders/val_loader.pkl', 'wb') as f:
    pickle.dump(val_loader, f)

