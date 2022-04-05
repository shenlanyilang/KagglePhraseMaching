from sqlite3 import SQLITE_DROP_TEMP_TRIGGER
import pandas as pd
import numpy as np
from datasets import load_dataset, DatasetDict, concatenate_datasets
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
import copy


wandb.init(project='my-project', name='exp040301')
wandb.watch_called = False
config = wandb.config
config.batch_size = 16
config.valid_batch_size = 8
config.epochs = 7
config.lr = 2e-5
config.classifier_lr = 2e-5
config.seed = 200
config.log_interval = 10
config.eval_steps = 100

data_dir = '/data/gehl/data/playground'

def tokenize_function(example):
    res = tokenizer(example['text1'], example['text2'], truncation=True)
    return res

# def tokenize_function_reverse(example):
#     res = tokenizer(example['title_target'], example['anchor'], truncation=True)
#     return res

# def concate_title(example):
#     example['title_anchor'] = example['title'] + ' ' + example['anchor']
#     return example

def concate_title_v2(example):
    example['text1'] = example['title'] + '[SEP]' + example['anchor']
    example['text2'] = example['target']
    return example

def concate_title_v2_reverse(example):
    example['text1'] = example['title'] + '[SEP]' + example['target'] 
    example['text2'] = example['anchor']
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

def get_valid_result_v2(model, dataloader):
    model.eval()
    device = model.device
    preds = np.array([])
    labels = np.array([])
    loss = 0
    for batch in dataloader:
        inputs = {k:v.to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        loss += outputs.loss.item()
        # pred_labels = torch.argmax(logits, dim=1).cpu().numpy()
        preds = np.append(preds, logits.flatten().cpu().numpy())
        labels = np.append(labels, batch['labels'].cpu().numpy())
    pear_score = pearsonr(preds, labels)[0]
    avg_loss = loss / len(dataloader)
    # metrics = compute_metrics(preds, labels)
    return {'pearsonr correlation': pear_score, 'loss': avg_loss}

def get_preds_labels(model, dataloader):
    model.eval()
    device = model.device
    preds = np.array([])
    labels = np.array([])
    loss = 0
    for batch in dataloader:
        inputs = {k:v.to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        loss += outputs.loss.item()
        # pred_labels = torch.argmax(logits, dim=1).cpu().numpy()
        preds = np.append(preds, logits.flatten().cpu().numpy())
        labels = np.append(labels, batch['labels'].cpu().numpy())
    return labels, preds, loss / len(dataloader)
    
def get_valid_result_v3(model, dataloader, dataloader_reverse):
    labels, preds, loss = get_preds_labels(model, dataloader)
    labels_reverse, preds_reverse, loss_reverse = get_preds_labels(model, dataloader_reverse)
    assert all(labels == labels_reverse)
    preds_combine = (preds + preds_reverse) / 2
    pear_score = pearsonr(preds_combine, labels)[0]
    avg_loss = (loss + loss_reverse) / 2
    # metrics = compute_metrics(preds, labels)
    return {'pearsonr correlation': pear_score, 'loss': avg_loss}
    

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

    # ds_dict = total_ds['train'].train_test_split(test_size=0.2, seed=config.seed)
    # train_ds, valid_test_ds = ds_dict['train'], ds_dict['test']
    # ds_dict = valid_test_ds.train_test_split(test_size=0.5, seed=config.seed)
    # valid_ds, test_ds = ds_dict['train'], ds_dict['test']
    # total_ds_dict = DatasetDict({'train': train_ds, 'valid':valid_ds, 'test': test_ds})
    # total_ds_dict.save_to_disk('/data/gehl/data/playground/us-patent-phrase-dataset.json')


    classifier_params_names = ['classifier.weight', 'classifier.bias']
    total_ds_dict = DatasetDict.load_from_disk('/data/gehl/data/playground/us-patent-phrase-dataset.json')
    train_ds, valid_ds, test_ds = total_ds_dict['train'], total_ds_dict['valid'], total_ds_dict['test']
    train_ds_reverse = copy.deepcopy(train_ds)
    valid_ds_reverse = copy.deepcopy(valid_ds)
    train_ds = train_ds.map(concate_title_v2)
    train_ds_reverse = train_ds_reverse.map(concate_title_v2_reverse)
    train_ds = concatenate_datasets([train_ds, train_ds_reverse]).shuffle(seed=config.seed)
    valid_ds = valid_ds.map(concate_title_v2)
    valid_ds_reverse = valid_ds_reverse.map(concate_title_v2_reverse)

    tokenizer = AutoTokenizer.from_pretrained('anferico/bert-for-patents')
    tokenized_train_ds = train_ds.map(tokenize_function, batched=True)
    tokenized_valid_ds = valid_ds.map(tokenize_function, batched=True)
    tokenized_valid_ds_reverse = valid_ds_reverse.map(tokenize_function, batched=True)
    tokenized_train_ds.rename_column_('score', 'labels')
    tokenized_valid_ds.rename_column_('score', 'labels')
    tokenized_valid_ds_reverse.rename_column_('score', 'labels')
    # tokenized_train_ds.column_names
    tokenized_train_ds.remove_columns_(['anchor', 'code', 'id', 'target', 'title', 'class', 'text1', 'text2'])
    tokenized_valid_ds.remove_columns_(['anchor', 'code', 'id', 'target', 'title', 'class', 'text1', 'text2'])
    tokenized_valid_ds_reverse.remove_columns_(['anchor', 'code', 'id', 'target', 'title', 'class', 'text1', 'text2'])
    # tokenized_train_ds.column_names

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    tokenized_train_ds.set_format('torch')
    tokenized_valid_ds.set_format('torch')
    tokenized_valid_ds_reverse.set_format('torch')

    train_dataloader = DataLoader(tokenized_train_ds, batch_size=config.batch_size, collate_fn=data_collator)
    valid_dataloader = DataLoader(tokenized_valid_ds, batch_size=config.valid_batch_size, collate_fn=data_collator)
    valid_dataloader_reverse = DataLoader(tokenized_valid_ds_reverse, batch_size=config.valid_batch_size, collate_fn=data_collator)

    print('dataloader generated successfully')

    model = AutoModelForSequenceClassification.from_pretrained('anferico/bert-for-patents', num_labels=1)
    
    device = torch.device('cuda')
    optimizer = AdamW(params=[
        {'params':(param for name, param in model.named_parameters() if name in classifier_params_names), 'lr': config.classifier_lr},
        {'params':(param for name, param in model.named_parameters() if name not in classifier_params_names), 'lr': config.lr}
        ])
    # optimizer = AdamW(params=model.parameters(), lr=config.lr)
    lr_scheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=int(0.05*len(train_dataloader)*config.epochs), num_training_steps=len(train_dataloader)*config.epochs)

    wandb.watch(model, log='all')
    wandb.define_metric('custom_step')
    wandb.define_metric('train loss', step_metric='custom_step')
    wandb.define_metric('valid loss', step_metric='custom_step')
    wandb.define_metric('valid pearsonr', step_metric='custom_step')

    total_steps = len(train_dataloader) * config.epochs

    model.to(device)
    step = 0
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
            step += 1
            if step % config.log_interval == 0:
                wandb.log({'train loss': loss.item(),
                           'custom_step': step})
            if step % config.eval_steps == 0:
                valid_metrics = get_valid_result_v3(model, valid_dataloader, valid_dataloader_reverse)
                wandb.log({'valid pearsonr': valid_metrics['pearsonr correlation'],
                        'valid loss': valid_metrics['loss'],
                        'custom_step': step})
                # print('valid metrics : {}'.format(valid_metrics))
    torch.save(model.state_dict(), '/data/gehl/data/playground/models/ussp_exp040301.pth')