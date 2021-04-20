import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import numpy as np
from pre_processing import normalization,tokenization,stemming,bag_of_words
from model import LSTM

with open('data3.json', 'r') as instances:
    data=json.load(instances)

word_list=[] #create an array with all the words
tags=[]
xy=[]
for i in data['data']:
    tag=i['tag']
    tags.append(tag)
    for user_response in i['user_responses']:
        normalized=normalization(user_response)
        words=tokenization(normalized)
        word_list.extend(words)# not apeend becous dont want a arraylist in a array
        xy.append((words,tag)) #array of user responses with the respective tags

word_list=[stemming(word)for word in word_list] # to remove the symbols
word_list=sorted(set(word_list)) # to remove duplicate elements
print(tags)
print(word_list)
print(xy)
# train the data
x_train=[]
y_train=[]
for(tokenized,tag) in xy:
    bag=bag_of_words(tokenized,word_list)
    x_train.append(bag)

    tag_label=tags.index(tag)
    y_train.append(tag_label)

x_train=np.array(x_train)
y_train=np.array(y_train)
print(bag)
print(x_train)
print(y_train)

inputs=torch.Tensor(x_train)
labels=torch.LongTensor(y_train)

class chatData(Dataset):
    def __init__(self):
        self.n_samples=len(x_train)
        self.x_data=inputs
        self.y_data=labels

    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.n_samples

batch_size=2
hidden_size=4
output_size=len(tags)
input_size=len(inputs[0])
seq_length=1
num_layers=2
num_classes=24
dataset=chatData()

train_loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
device = torch.device("cpu")
model=LSTM(seq_length,input_size,hidden_size,num_layers,num_classes).to(device)

num_epochs=4000
learning_rate=0.001

#Loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

#Train model
for epoch in range(num_epochs):
    for (inputs,labels) in train_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        output,hidden=model(inputs)
        loss=criterion(output,labels)
        optimizer.zero_grad() # Clears existing gradients from previous epoch
        loss.backward()
        optimizer.step()

    if(epoch+1)%100==0:
       print(f'epoch {epoch+1}/{num_epochs}')
       print(f'loss={loss.item():.4f}')

data={
    "model_state":model.state_dict(),
    "seq_length":seq_length,
    "input_size":input_size,
    "hidden_size":hidden_size,
    "num_layers":num_layers,
    "num_classes":num_classes,
    "word_list":word_list,
    "tags":tags
}

FILE="dataserialized.pth"
torch.save(data,FILE)
print("Training complete, file saved to dataserialized.pth")
