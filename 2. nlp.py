import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import os
import re
import numpy as np
import pandas as pd
import torch

def get_data():
    pos1, pos2 = os.listdir('./test/pos'), os.listdir('./train/pos')
    neg1, neg2 = os.listdir('./test/neg'), os.listdir('./train/neg')
    pos_all, neg_all = [], []
    for p1, n1 in zip(pos1, neg1):
        with open('./test/pos/' + p1, encoding='utf8') as f:
            pos_all.append(f.read())
        with open('./test/neg/' + n1, encoding='utf8') as f:
            neg_all.append(f.read())
    for p2, n2 in zip(pos2, neg2):
        with open('./train/pos/' + p2, encoding='utf8') as f:
            pos_all.append(f.read())
        with open('./train/neg/' + n2, encoding='utf8') as f:
            neg_all.append(f.read())
    datasets = np.array(pos_all + neg_all)
    labels = np.array([1] * 25000 + [0] * 25000)
    return datasets, labels

def shuffle_process():
    sentences, labels = get_data()
    # Shuffle
    shuffle_indexs = np.random.permutation(len(sentences))
    datasets = sentences[shuffle_indexs]
    labels = labels[shuffle_indexs]
    return datasets,labels

def save_process():
    datasets, labels = shuffle_process()
    sentences = []
    punc = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]'
    for sen in datasets:
        sen = sen.replace('\n', '')
        sen = sen.replace('<br /><br />', ' ')
        sen = re.sub(punc, '', sen)
        sentences.append(sen)
    return pd.DataFrame({'labels': labels, 'sentences': sentences})

class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}

def train_model(df_pred):
    pred_texts = df_pred["sentences"].dropna().astype('str').tolist()

    # Load tokenizer and model, create trainer
    model_name = "siebert/sentiment-roberta-large-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    trainer = Trainer(model=model)

    # Tokenize texts and create prediction data set
    tokenized_texts = tokenizer(pred_texts, truncation=True,padding=True)
    pred_dataset = SimpleDataset(tokenized_texts)

    # Run predictions
    predictions = trainer.predict(pred_dataset)

    # Transform predictions to labels
    preds = predictions.predictions.argmax(-1)
    labels = pd.Series(preds).map(model.config.id2label)
    scores = (np.exp(predictions[0])/np.exp(predictions[0]).sum(-1,keepdims=True)).max(1)

    # Create DataFrame with texts, predictions, labels, and scores
    df = pd.DataFrame(list(zip(pred_texts,preds,labels,scores)), columns=['text','pred','label','score'])
    return df

def main():
    df_pred = save_process()
    df = train_model(df_pred)
    df.to_csv (r'outcome.csv', index = None)

if __name__ == '__main__':
    main()