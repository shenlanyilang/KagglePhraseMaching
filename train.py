import pandas as pd
import numpy as np
from datasets import load_dataset, DatasetDict
from os import path
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import AdamW
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from transformers import get_scheduler
import logging
logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)
import wandb


wandb.init(project='my-project', name='exp040202')
wandb.watch_called = False
config = wandb.config
config.batch_size = 16
config.epochs = 10
config.lr = 2e-5
config.seed = 200
config.log_interval = 10
config.eval_steps = 100

data_dir = '/data/gehl/data/playground'

def tokenize_function(example):
    res = tokenizer(example['title_anchor'], example['target'], truncation=True)
    return res

def concate_title(example):
    example['title_anchor'] = example['title'] + ' ' + example['anchor']
    return example

label_score_mapping = {0: 0, 1:0.25, 2:0.5, 3:0.75, 4:1.}

def compute_metrics(preds, labels):
#     print(preds[:20])
#     print(labels[:20])
    preds_scores = np.array([label_score_mapping[lab] for lab in preds])
    labels_scores = np.array([label_score_mapping[lab] for lab in labels])
#     print(preds_scores[:20])
#     print(labels_scores[:20])
    r = pearsonr(preds_scores, labels_scores)[0]
    acc = accuracy_score(labels, preds)
    return {'pearson correlation':r, 'label accuracy': acc}

def get_valid_result(model, dataloader):
    model.eval()
    device = model.device
    preds = np.array([])
    labels = np.array([])
    for batch in dataloader:
        inputs = {k:v.to(device) for k,v in batch.items() if k != 'labels'}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        pred_labels = torch.argmax(logits, dim=1).cpu().numpy()
        preds = np.append(preds, pred_labels)
        labels = np.append(labels, batch['labels'].cpu().numpy())
    metrics = compute_metrics(preds, labels)
    return metrics

# from tqdm.notebook import tqdm


if __name__ == '__main__':
    # total_ds = load_dataset('json', data_files=path.join(data_dir, 'us-patent-dataset.json'))
    # total_ds = total_ds.map(concate_title)

    # ds_dict = total_ds['train'].train_test_split(test_size=0.2, seed=config.seed)
    # train_ds, valid_test_ds = ds_dict['train'], ds_dict['test']
    # ds_dict = valid_test_ds.train_test_split(test_size=0.5, seed=config.seed)
    # valid_ds, test_ds = ds_dict['train'], ds_dict['test']
    total_ds_dict = DatasetDict.load_from_disk('/data/gehl/data/playground/us-patent-phrase-dataset.json')
    train_ds, valid_ds, test_ds = total_ds_dict['train'], total_ds_dict['valid'], total_ds_dict['test']

    tokenizer = AutoTokenizer.from_pretrained('anferico/bert-for-patents')
    tokenized_train_ds = train_ds.map(tokenize_function, batched=True)
    tokenized_valid_ds = valid_ds.map(tokenize_function, batched=True)
    tokenized_train_ds.rename_column_('class', 'labels')
    tokenized_valid_ds.rename_column_('class', 'labels')
    # tokenized_train_ds.column_names
    tokenized_train_ds.remove_columns_(['anchor', 'code', 'id','score', 'target', 'title', 'title_anchor'])
    tokenized_valid_ds.remove_columns_(['anchor', 'code', 'id','score', 'target', 'title', 'title_anchor'])
    # tokenized_train_ds.column_names

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    tokenized_train_ds.set_format('torch')
    tokenized_valid_ds.set_format('torch')

    train_dataloader = DataLoader(tokenized_train_ds, batch_size=config.batch_size, collate_fn=data_collator)
    valid_dataloader = DataLoader(tokenized_valid_ds, batch_size=config.batch_size, collate_fn=data_collator)
    print('dataloader generated successfully')

    model = AutoModelForSequenceClassification.from_pretrained('anferico/bert-for-patents', num_labels=5)
    
    device = torch.device('cuda')
    optimizer = AdamW(params=model.parameters(), lr=config.lr)
    lr_scheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*config.epochs)

    wandb.watch(model, log='all')

    total_steps = len(train_dataloader) * config.epochs

    model.to(device)
    step = 1
    # pgbar = tqdm(total=total_steps, desc='train loss:')
    for epo in range(config.epochs):
        for batch in train_dataloader:
            model.train()
            inputs = {k:v.to(device) for k,v in batch.items()}
    #         print(inputs)
    #         break
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if step % config.log_interval == 0:
                wandb.log({'train loss': loss.item()})
            step += 1
            if step % config.eval_steps == 0:
                valid_metrics = get_valid_result(model, valid_dataloader)
                wandb.log({'valid pearsonr': valid_metrics['pearson correlation'],
                        'valid acc': valid_metrics['label accuracy']})
                # print('valid metrics : {}'.format(valid_metrics))
    torch.save(model.state_dict(), '/data/gehl/data/playground/models/ussp_exp040202.pth')