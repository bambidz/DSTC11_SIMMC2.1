import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForMaskedLM, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EvalPrediction

tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

special_tokens = {}
additional_special_tokens = ["<MO>", "<NMO>", "<UT>", "<ST>", "<P1>", "<P2>"]
special_tokens["additional_special_tokens"] = additional_special_tokens

num_added_toks = tokenizer.add_special_tokens(special_tokens)

class subtask2_dataset(Dataset):
    def __init__(self, path):
        with open(path) as f: # "../data/object_f1_train_10.txt"
            self.data = f.readlines()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        split_data = self.data[idx].split("\t")
        
        # Hard Prompt <s1> ? <mask> , <s2>       https://gaotianyu.xyz/prompting/
        text = split_data[0] + " " + " <P1><P2><mask> , " + split_data[5].strip() + " " + split_data[1]
        
        if split_data[2] == "1":
            label_1 = split_data[0] + " " + " <P1><P2>Yes , " + split_data[5].strip() + " " + split_data[1] # " Yes" : 3216
        else:
            label_1 = split_data[0] + " " + " <P1><P2>No , " + split_data[5].strip() + " " + split_data[1] # " No" : 440
        
#        return text, label_1, label_2
        return text, label_1

class subtask2_RoBERTa(nn.Module):
    def __init__(self):
        super(subtask2_RoBERTa, self).__init__()
        self.encoder = RobertaForMaskedLM.from_pretrained("roberta-large")
        self.encoder.resize_token_embeddings(len(tokenizer))

#    def forward(self, ids, mask, label1, label2):
    def forward(self, ids, mask, label1):
        encoder_output = self.encoder(
            ids, 
            labels = label1,
            attention_mask=mask,
        )
        
        loss = encoder_output.loss
        logit = encoder_output.logits
        
        labels = label1[label1!=-100]
        logits = logit[label1!=-100]
        
        y_logits = logits[:, 9904].data
        n_logits = logits[:, 3084].data
        
        # Yes index / "Yes", " Yes"
        labels[labels==9904] = 1
        # labels[labels==3216] = 1
        
        # No index / "No", " No"
        labels[labels==3084] = 0
        # labels[labels==440] = 0
        
        return {'yes': y_logits, 'no': n_logits, 'label': labels, 'loss': loss}      

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        label1 = inputs.get("label1").to(device)
        # label2 = inputs.get("label2").to(device)
        
        model_inputs = {'input_ids':inputs['input_ids'].to(device), 'attention_mask':inputs['attention_mask'].to(device)}

        # forward pass
        # outputs = model(model_inputs['input_ids'], model_inputs['attention_mask'], label1, label2)
        outputs = model(model_inputs['input_ids'], model_inputs['attention_mask'], label1)
        
        loss = outputs["loss"]
        
        return (loss, outputs) if return_outputs else loss
    
def compute_metrics(p: EvalPrediction):
 
    y_preds = p.predictions[0]
    n_preds = p.predictions[1]
    label = p.predictions[2]
    
#    print(y_preds)
#    print(n_preds)
#    print(label)
    
    t = time.time()
    
    try:
        with open("logs/large_mo_turn2_softprompt2_v2_pred_1e-5.json") as f:
            json_result = json.load(f)
    except:
        json_result = {}
        
    json_result[str(t)] = {'y_logit': y_preds.tolist(), 'n_logit': n_preds.tolist(), 'label': label.tolist()}

    with open("logs/large_mo_turn2_softprompt2_v2_pred_1e-5.json", "w") as f:
        json.dump(json_result, f, indent=4)

        
    preds = np.zeros(y_preds.shape)
    preds[y_preds>n_preds] = 1.0 
        
    precision1, recall1, f11, _ = precision_recall_fscore_support(label, preds, average='binary')
    # precision2, recall2, f12, _ = precision_recall_fscore_support(label2, preds2, average='binary')
    
    acc1 = accuracy_score(label, preds)
    # acc2 = accuracy_score(label2, preds2)

    return {
        't1_acc': acc1, 't1_f1': f11, 't1_p': precision1, 't1_r': recall1
    }



def data_collator(data):
    text = [data_i[0] for data_i in data]
    label_1 = [data_i[1] for data_i in data]
    
    encoded_inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", add_special_tokens=True)
    labels = tokenizer(label_1, padding=True, truncation=True, return_tensors="pt", add_special_tokens=True)["input_ids"]
    
    labels = torch.where(encoded_inputs.input_ids == tokenizer.mask_token_id, labels, -100)
    
    encoded_inputs['label1'] = labels

    return encoded_inputs
    

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(0)


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device('cpu')

train_data = subtask2_dataset("./data/object_f1_train_turn2.txt")
test_data = subtask2_dataset("./data/object_f1_devtest_turn2_with_pred.txt")
#test_data, _ = random_split(test_data, [100, len(test_data) - 100])

model = subtask2_RoBERTa()

model.to(device)

training_args = TrainingArguments(
    output_dir = './save_model',
    num_train_epochs = 3,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 16,    
    per_device_eval_batch_size = 8,
    learning_rate = 1e-5,
    evaluation_strategy = "steps",
    disable_tqdm = False, 
    load_best_model_at_end = True,
    save_steps = 2000,
    warmup_steps = 500,
    weight_decay = 0.01,
    logging_steps = 2000,
    fp16 = True,
    logging_dir = './logs',
    dataloader_num_workers = 32,
    run_name = 'roberta-classification'
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

trainer.train()
trainer.evaluate()
