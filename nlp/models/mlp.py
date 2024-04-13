import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

from copy import deepcopy
import random
import warnings
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn import Linear
from torch.nn.functional import dropout
from torch.utils.data import Dataset, DataLoader

WV_SIZE = 300
N_CLASSES = 4

###############################
########## MLP MODEL ##########
###############################

class MLP(torch.nn.Module):
    def __init__(self, nHiddenChannels1, nHiddenChannels2):
        super().__init__()
        self.lin1 = Linear(WV_SIZE, nHiddenChannels1)
        self.lin2 = Linear(nHiddenChannels1, nHiddenChannels2)
        self.lin3 = Linear(nHiddenChannels2, N_CLASSES)
        
    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = x.relu()
        x = self.lin3(x)
        
        x = dropout(x, p=0.5, training=self.training)
        return x
    
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, ind):
        x = self.data[ind]
        y = self.labels[ind]
        return x, y
    
###############################
######### MAIN METHOD #########
###############################
    
def run_mlp(nEpochs=15, lr=0.00005):
    warnings.filterwarnings("ignore")
    
    # Initialise
    model = MLP(nHiddenChannels1=800, nHiddenChannels2=400) # set number of hidden nodes in each layer here
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    softmax = torch.nn.Softmax(dim=0)
    
    train_data = pd.read_pickle('nlp/models/dataset/way11/way11_train_doc.pkl')
    test_data = pd.read_pickle('nlp/models/dataset/way11/way11_test_doc.pkl')
        
    freqCutOff = int(len(train_data['embeddings'])*0.8)
    x_combined, x_test = [[j for j in i] for i in train_data['embeddings']], [[j for j in i] for i in test_data['embeddings']]      
    y_combined, y_test = [int(i)-1 for i in train_data['Label']], [int(i)-1 for i in test_data['Label']]
    c = list(zip(x_combined, y_combined))
    random.shuffle(c)
    x_combined, y_combined = zip(*c)
    x_train, x_val = x_combined[:freqCutOff], x_combined[freqCutOff:]
    y_train, y_val = y_combined[:freqCutOff], y_combined[freqCutOff:]

    # train_data, test_data = pd.read_csv('nlp/models/way1_train.csv').sample(frac=1), pd.read_csv('nlp/models/way1_test.csv')
    # freqCutOff = int(len(train_data['text_lowercase'])*0.8)
    # x_train, x_val, x_test = train_data['text_lowercase'][:freqCutOff], train_data['text_lowercase'][freqCutOff:], test_data['text_lowercase']
    # x_train, dim = embed.get_embedding(x_train); print("x_train processing done");
    # x_val, dim = embed.get_embedding(x_val); print("x_val processing done");
    # x_test, dim = embed.get_embedding(x_test); print("x_test processing done");
    # y_train, y_val, y_test = [i-1 for i in train_data['Label'][:freqCutOff]], [i-1 for i in train_data['Label'][freqCutOff:]], [i-1 for i in test_data['Label']]
    
    # Helper functions for training, testing, and generating word vectors
    def train(loader):
        model.train()
        losses = []
        for data in loader:
            x, y = data[0], data[1]

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        return losses
    
    def test(loader):
        model.eval()
        pred, probs, labels = [], [], []
        for data in loader:
            x, y = data[0], data[1]
            
            out = model(x)
            for i in out:
                probs.append(softmax(i).tolist())
                pred.append(torch.argmax(i).item())
            labels.extend(y)
            
        with torch.no_grad():
            f1, precision, recall, acc, roc = f1_score(labels, pred, average='macro'), precision_score(labels, pred, average='macro'), recall_score(labels, pred, average='macro'), accuracy_score(labels, pred), roc_auc_score(labels, probs, average='macro', multi_class='ovr')
        return f1, precision, recall, acc, roc
    
    print("Starting training...")
    
    losses = []
    f1s, precisions, recalls, accs, rocs = [], [], [], [], []
    
    val_dataset = CustomDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    for i in range(nEpochs):
        curr_xtrain, curr_ytrain = deepcopy(x_train), deepcopy(y_train)
        c = list(zip(curr_xtrain, curr_ytrain))
        random.shuffle(c)
        curr_xtrain, curr_ytrain = zip(*c)
        train_dataset = CustomDataset(torch.tensor(curr_xtrain, dtype=torch.float32), torch.tensor(curr_ytrain, dtype=torch.long))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        currLosses = train(train_loader)
        losses.extend(currLosses)
        minLoss, avgLoss = min(losses), sum(losses)/len(losses)

        f1, precision, recall, acc, roc = test(val_loader)
        f1s.append(f1); precisions.append(precision); recalls.append(recall); accs.append(acc); rocs.append(roc);
        
        print(f'''Val Scores.
                Epoch: {i} | Min Train Loss: {minLoss} | Avg Train Loss: {avgLoss}
                Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | ROC: {roc:.4f}''')
    
    # Generate prediction on test data
    test_dataset = CustomDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    f1, precision, recall, acc, roc = test(test_loader)
    print(f'''Test Scores.
            Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | ROC: {roc:.4f}''')    
    
    # Visualisation of loss.
    losses_float = [float(loss) for loss in losses] 
    loss_indices = [i for i,l in enumerate(losses_float)] 
    d = {'indices': np.array(loss_indices), 'loss': np.array(losses_float)}
    pdnumsqr = pd.DataFrame(d)
    sns.lineplot(x='indices', y='value', hue='variable', data=pd.melt(pdnumsqr, ['indices']))
    # plt.ylim(0, 20)
    plt.show()
    
    # Visualisation of performance
    epoch = [i for i in range(nEpochs)] 
    d = {'epoch': np.array(epoch), 'ROC': np.array(rocs), 'F1': np.array(f1s), 'Precision': np.array(precisions), 'Recall': np.array(recalls), 'Accuracy': np.array(accs)}
    pdnumsqr = pd.DataFrame(d)
    sns.lineplot(x='epoch', y='value', hue='variable', data=pd.melt(pdnumsqr, ['epoch']))
    plt.show()