import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score

import warnings
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from nlp.embedding.embed import Embedding
from nlp.models.mlp import MLP, run_mlp
from nlp.models.lstm import LSTM, run_lstm

N_CLASSES = 4

def main(nEpochs=5, lr=0.00005):
    warnings.filterwarnings("ignore")
    
    # Initialise
    model = MLP(nHiddenChannels1=800, nHiddenChannels2=400) # set number of hidden nodes in each layer here
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    embed = Embedding(type='fasttext')
    train_data, test_data = pd.read_csv('nlp/models/way1_train.csv'), pd.read_csv('nlp/models/way1_test.csv')
    freqCutOff = int(len(train_data['text_lowercase'])*0.8)
        
    # Get word vectors for each unique token
    x, dim = embed.get_embedding(train_data['text_lowercase'])
    x_train = x[:freqCutOff]
    x_val = x[freqCutOff:]
    x_test, dim = embed.get_embedding(test_data['text_lowercase'])

    # no. cols * 300, no. cols * (100 * 400)
    
    # x_train, x_val, x_test = [[sum(x) for x in zip(*embed.get_embedding(i.split()))] for i in x_train], [[sum(x) for x in zip(*embed.get_embedding(i.split()))] for i in x_val], [[sum(x) for x in zip(*embed.get_embedding(i.split()))] for i in x_test]
    y_train, y_val, y_test = [i-1 for i in train_data['Label'][:freqCutOff]], [i-1 for i in train_data['Label'][freqCutOff:]], [i-1 for i in test_data['Label']]
    
    # Helper functions for training, testing, and generating word vectors
    def train(x_train, y_train):
        model.train()
        losses = []
        for i in range(len(x_train)):
            optimizer.zero_grad()
            
            x, y = torch.tensor(x_train[i], dtype=torch.float32), torch.tensor([y_train[i]], dtype=torch.int64)
            out = model(x)
            out = torch.reshape(out, (1,N_CLASSES))
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        return losses
    
    def test(x_test):
        model.eval()
        res = []
        for i in x_test:
            out = torch.argmax(model(torch.tensor(i, dtype=torch.float32))).item()
            res.append(out)

        return res
    
    print("Starting training...")
    
    losses = []
    f1s, precisions, recalls, accs, rocs, prcs = [], [], [], [], [], []
    rocs, prcs = [], []
    for i in range(nEpochs):
        currLosses = train(x_train, y_train)
        losses.extend(currLosses)
        minLoss, avgLoss = min(losses), sum(losses)/len(losses)

        y_pred = test(x_val)
        f1, precision, recall, acc, roc, prc = f1_score(y_val, y_pred, average='macro'), precision_score(y_val, y_pred, average='macro'), recall_score(y_val, y_pred, average='macro'), accuracy_score(y_val, y_pred), roc_auc_score(y_val, y_pred, average='macro'), average_precision_score(y_val, y_pred, average='macro')
        f1s.append(f1); precisions.append(precision); recalls.append(recall); accs.append(acc); rocs.append(roc); prcs.append(prc)
        
        print(f'''Val Scores.
                Epoch: {i} | Min Train Loss: {minLoss} | Avg Train Loss: {avgLoss}
                Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | ROC: {roc:.4f} | PRC: {prc:.4f}''')
        
    # Visualisation of loss.
    losses_float = [float(loss) for loss in losses] 
    loss_indices = [i for i,l in enumerate(losses_float)] 
    d = {'indices': np.array(loss_indices), 'loss': np.array(losses_float)}
    pdnumsqr = pd.DataFrame(d)
    sns.lineplot(x='indices', y='value', hue='variable', data=pd.melt(pdnumsqr, ['indices']))
    plt.ylim(0, 20)
    plt.show()
    
    # Visualisation of performance
    epoch = [i for i in range(nEpochs)] 
    d = {'epoch': np.array(epoch), 'ROC': np.array(rocs), 'PRC': np.array(prcs), 'F1': np.array(f1s), 'Precision': np.array(precisions), 'Recall': np.array(recalls), 'Accuracy': np.array(accs)}
    pdnumsqr = pd.DataFrame(d)
    sns.lineplot(x='epoch', y='value', hue='variable', data=pd.melt(pdnumsqr, ['epoch']))
    plt.show()
        
    # Generate prediction on test data
    y_pred = test(x_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    print(f'''Test Scores.
            Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} ''')


if __name__ == "__main__":
    # main()
    # run_mlp()
    run_lstm()
