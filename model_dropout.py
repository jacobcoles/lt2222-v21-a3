import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random

class VowelModel(nn.Module):
    def __init__(self, featuresize, hiddensize, outputsize, vocab):
        super().__init__()
        self.lin1 = nn.Linear(featuresize, hiddensize)
        self.tanh = nn.Tanh()
        self.lin2 = nn.Linear(hiddensize, hiddensize)
        self.sigmoid = nn.Sigmoid()
        self.lin3 = nn.Linear(hiddensize, outputsize)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=0.2)
        self.vocab = vocab

    def forward(self, x):
        x = self.lin1(x)
        x = self.tanh(x)
        x = self.lin2(x)
        x = self.sigmoid(x)
        x = self.lin3(x)
        x = self.softmax(x)
        x = self.dropout(x)
        return x

def __shuffler__(X, y):
    df = pd.DataFrame(X)
    df['class'] = y
    indices = list(range(X.shape[0]))
    
    while True:
        random.shuffle(indices)
        rearrangedf = df.iloc[indices[:int(len(indices)/2)]]
        featuredf = rearrangedf.drop('class', axis=1)
        classdf = rearrangedf['class']

        yield torch.Tensor(featuredf.to_numpy()), torch.LongTensor(classdf.to_numpy())
        

def train_dropout(X, y, vocab, hiddensize, epochs=100):
    inputsize = X.shape[1]
    outputsize = int(y.max()) + 1
    model = VowelModel(inputsize, hiddensize, outputsize, vocab)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    shuffler = __shuffler__(X, y)

    for epoch in range(epochs):
        Xt, yt = next(shuffler)
        
        optimizer.zero_grad()
        outputs = model(Xt.unsqueeze(0))
        loss = criterion(outputs.squeeze(0), yt)
        loss.backward()
        optimizer.step()

        print("In epoch {}, the loss was {}.".format(epoch, loss))

    return model

 

