
# coding: utf-8

# In[ ]:


"""TED Talk Generator training code
Character-based language model trained on TED Talks"""

# Author: Sameera Lanka
# Website: www.sameera-lanka.com
# Copyright © TED Conferences, LLC

import glob
import os
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

#__________Data loading__________#

corpus = ""
for filename in glob.glob('TED_transcripts/*.txt'):
    with open(filename ,'r') as f:
        corpus += 2*'\n' + f.read().replace('\n', ' ')

train_split = 0.98   # fraction of training data 
train_size = int((train_split)*len(corpus))

train_data = corpus[:train_size]
val_data = corpus[train_size:]

#__________Parameters____________#

vocab = list(set(corpus))
ip_dim = len(vocab)  # one-hot dimension of char embedding
hi_dim = 128         # no. of hidden units
sample_size = 128    # input text snippet size
batch_size = 64      # mini-batch size
num_layers = 2       # no.of lstm layers    
num_epochs = 100      # no. of complete passes through corpus
learning_rate = 1e-5 
onehot_mat = torch.eye(ip_dim)  # matrix for one-hot lookup
char2oh = {vocab[i]:onehot_mat[i] for i in range(ip_dim)}  


# In[ ]:


#_________Building LSTM_________#

class network(nn.Module):
    def __init__(self, ip_dim, hi_dim, num_layers):    
        super(network, self).__init__()
        self.ip_dim = ip_dim
        self.hi_dim = hi_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
    
        self.lstm = nn.LSTM(self.ip_dim, self.hi_dim, self.num_layers, batch_first=True)
        self.linear = nn.Linear(hi_dim, ip_dim) 
        
    def reset(self):
        self.h0 = Variable(torch.zeros(self.num_layers, self.batch_size, self.hi_dim)).cuda()
        self.c0 = Variable(torch.zeros(self.num_layers, self.batch_size, self.hi_dim)).cuda()
        self.hidden = self.h0, self.c0
    
    def forward(self, ip):
        op_pred, self.hidden = self.lstm(ip, self.hidden)
        op_pred = self.linear(op_pred).view(-1, ip_dim)
        return op_pred


# In[ ]:


#_______Support Functions_______#

def random_sample(data):
    """Generate a random text snippet and label containing the corresponding next letters"""
    start = np.random.randint(0, len(data) - sample_size)
    end = start + sample_size 
    sample = data[start:end]
    label = data[start+1:end+1]
    data_reduced = data.replace(sample, "")
    return sample, label, data_reduced


def genbatch(dataset):
    """Generate a unique batch of text samples in randomized order from the dataset"""
    ip = torch.zeros(batch_size, sample_size, ip_dim)
    target = torch.zeros(batch_size, sample_size).type(torch.LongTensor)
    for b in range(batch_size):
        ip_sample, target_sample, dataset = random_sample(dataset)
        for letter in range(sample_size):
            ip[b, letter,:] = char2oh[ip_sample[letter]]
            target[b, letter] = vocab.index(target_sample[letter])
    
    return ip, target, dataset


def validate(model):
    """Returns validation loss"""
    val_copy = val_data
    val_loss = 0
    val_batch_num = 20 # number of mini-batches of validation set to evaluate loss
    for i in range(val_batch_num):
        model.reset()
        val_ip, val_target, val_copy = genbatch(val_copy)
        val_ip = Variable((val_ip), volatile=True).cuda()
        val_target = Variable((val_target), volatile=True).cuda().view(-1)
        val_pred = model(val_ip)
        val_loss += criterion(val_pred,val_target)
    val_loss = val_loss.data[0]/val_batch_num
    return val_loss


def save_checkpoint(model, val_loss, e, optimizer):
    """Saves model checkpoint. File name = TED_(validation loss)"""
    cp_name = 'e{}_{:.4f}.pth'.format(e+1,val_loss)
    cp_path = cp_dir + cp_name
    opt_path = opt_dir + cp_name
    torch.save(model.state_dict(), cp_path) #save model
    torch.save(optimizer.state_dict(), opt_path) #save optimizer


# In[ ]:


#__________Training___________#

if __name__=="__main__":

    TED = network(ip_dim, hi_dim, num_layers).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(TED.parameters(),lr=learning_rate)

    cp_dir = "./checkpoints/" #model checkpoints directory
    if not os.path.isdir(cp_dir):
        os.mkdir(cp_dir)
        
    opt_dir = "./opt/"        #optimizer checkpoints directory
    if not os.path.isdir(opt_dir):
        op_dir = "./opt/"


    for e in range(num_epochs):
        train_copy = train_data
        num_passes = 0
        while len(train_copy) > (sample_size * batch_size): 
            loss = 0
            num_passes += 1
            
            TED.reset()    
            ip, target, train_copy = genbatch(train_copy) 
            ip = Variable(ip).cuda()
            target = Variable(target).cuda().view(-1)
            pred = TED(ip)
            
            loss = criterion(pred,target)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
       
        val_loss = validate(TED)
        print('\nepoch: {0} \ntraining loss: {1}, '
                'validation loss: {2}\n'.format(e+1, loss.data[0], val_loss))
        save_checkpoint(TED, val_loss, e, optimizer)

