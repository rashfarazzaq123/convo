import torch
import torch.nn as nn

device = torch.device("cpu")

class RNN(nn.Module):
    def __init__(self,seq_length,input_size,hidden_size,num_layers,num_classes):
        super(RNN,self).__init__()
        self.seq_length=seq_length
        self.hidden_size=hidden_size
        self.input_size=input_size
        self.num_layers=num_layers
        self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
        self.softmax=nn.LogSoftmax(dim=1)
        self.connected=nn.Linear(hidden_size,num_classes)
    
    def forward(self,x):
        batch_size=x.size(0)
        hidden=self.init_hidden(batch_size)
        x = x.view(batch_size, self.seq_length, self.input_size)
        out,hidden=self.rnn(x,hidden)
        out=out.contiguous().view(-1,self.hidden_size)
        out=self.connected(out)
        return out,hidden
    
    def init_hidden(self,batch_size):
        hidden=torch.zeros(self.num_layers,batch_size,self.hidden_size).to(device)
        return hidden

    #TEST
