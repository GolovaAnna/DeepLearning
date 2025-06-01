import os
os.environ['WANDB_MODE'] = 'offline'
import wandb
import pandas as pd
import torch
import numpy as np

from configs import config
from transformer import EncoderDecoder
from preprocessing import preprocessing
from metrics import LabelSmoothingLoss
from optimizer import NoamOpt
from transformer import fit

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor
    DEVICE = torch.device('cuda')
else:
    from torch import FloatTensor, LongTensor
    DEVICE = torch.device('cpu')

np.random.seed(42)

print('aaa')


def main():
    # Один раз выполните при первой настройке
    wandb.login()

    # Запуск проекта и конфигурации
    wandb.init(project="transformer-summarizer", config={
        "epochs": config['epochs'],
        "batch_size": config['b_size_train'],
        "learning_rate": config['learning_rate'],
        "model": "EncoderDecoder",
        "d_model": config['d_model'],
        "n_heads": config['n_heads'],
    })

    # !wandb sync "D:/05_Attention/05_Attention/seminar/wandb/offline-run-20250526_162725-2622dbq9"
    # !wandb sync "D:/05_Attention/05_Attention/seminar/wandb/offline-run-*"

    data = pd.read_csv('D:/05_Attention/05_Attention/seminar/news.csv', delimiter=',') 
    train_iter, test_iter, word_field, embedding_layer = preprocessing(data, tokenize=None)
    vocab_size = len(word_field.vocab)

    model = EncoderDecoder(word_field, embedding=embedding_layer)
    model = model.to(DEVICE)

    # criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(DEVICE) заменили на...
    pad_idx = word_field.vocab.stoi[word_field.pad_token]
    criterion = LabelSmoothingLoss(vocab_size,  padding_idx=pad_idx).to(DEVICE)

    optimizer = NoamOpt(model)

    fit(model, criterion, optimizer, train_iter, epochs_count=30, val_iter=test_iter)

if __name__ == "__main__":
    main()    