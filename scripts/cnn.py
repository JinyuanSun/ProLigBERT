import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy.stats import pearsonr

# -------------------Load Data
train_df = pd.read_pickle("../data/train_data.pkl")
test_df = pd.read_pickle("../data/test_data.pkl")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = {'data':[],'targets':[]}
test_data = {'data':[],'targets':[]}
print('[INFO]: Loading train data!')
for index, row in train_df.iterrows():
    train_data['data'].append(np.outer(row['mol embed'], row['protein embed']).reshape(1,512,1280))
    train_data['targets'].append(np.log(row['Kd (nM)']))

print('[INFO]: Loading test data!')
for index, row in test_df.iterrows():
    test_data['data'].append(np.outer(row['mol embed'], row['protein embed']).reshape(1,512,1280))
    test_data['targets'].append(np.log(row['Kd (nM)']))
    
    
from scipy import stats
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
         
        self.conv1 = nn.Conv2d(1,3,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(3,5,5)
        self.fc1 = nn.Linear(5 * 125 * 317,360)
        self.fc2 = nn.Linear(360,180)
        self.fc3 = nn.Linear(180,1)
         
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,5 * 125 * 317)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
         
        return x

#net = Net()

num_epochs = 80
batch_size = 128
learning_rate = 0.0005

model = Net()
loss_function = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
#test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
model.to(device=device)
model.train()

def test():
    #model = torch.load('network_1_ep300.pth')
    model.eval()

    test_y = np.array([])
    pred_y = np.array([])

    ep_loss = np.array([])
    index = 0
    while index < len(test_data['data']) - batch_size:
        data = np.array(test_data['data'][index: index + batch_size])
        targets = np.array(test_data['targets'][index: index + batch_size])
        index += batch_size
        #records, targets = (torch.split(batch_data, [1792, 1], dim=1))
        #data = []
        #for record in records:
        #    record_outer = torch.Tensor.outer(record[0:512],record[512:1792]).detach().numpy().reshape(1,512,1280)
        #    data.append(record_outer)
        data = torch.tensor(data).to(device=device)
        
        test_y = np.concatenate((test_y, targets.ravel()))
        #data = data.to(device=device)
        targets = torch.tensor(targets).to(device=device)
        #print(type(targets))
        predicted_output = model(data.float())
        pred_y = np.concatenate((pred_y, predicted_output.to(device='cpu').detach().numpy().ravel()))
        loss = loss_function(predicted_output, targets.float())
        ep_loss = np.concatenate((ep_loss.ravel(), loss.to(device='cpu').detach().numpy().ravel()))
    
    data = np.array(test_data['data'][index: ])
    targets = np.array(test_data['targets'][index: ])
    data = torch.tensor(data).to(device=device)
    test_y = np.concatenate((test_y, targets.ravel()))
    #data = data.to(device=device)
    targets = torch.tensor(targets).to(device=device)
    predicted_output = model(data.float())
    pred_y = np.concatenate((pred_y, predicted_output.to(device='cpu').detach().numpy().ravel()))
    loss = loss_function(predicted_output, targets.float())
    ep_loss = np.concatenate((ep_loss.ravel(), loss.to(device='cpu').detach().numpy().ravel()))
    #r, _pv = stats.pearsonr(test_y,pred_y)
    #print(f'Validation completed with Loss {ep_loss}')
    #print(f'Validation r: {r}')
    return np.mean(ep_loss.ravel()), test_y, pred_y


def train():
    train_loss = np.array([])
    test_loss = np.array([])
    for epoch in range(num_epochs):
        train_ep_loss = np.array([])
        # train_ep_loss = np.array([])
        index = 0
        
        while index < len(train_data['data']) - batch_size:
            data = np.array(train_data['data'][index: index + batch_size])
            targets = np.array(train_data['targets'][index: index + batch_size])
            index += batch_size
            #records, targets = (torch.split(batch_data, [1792, 1], dim=1))
            #data = []
            #for record in records:
            # record_outer = torch.Tensor.outer(record[0:512],record[512:1792]).detach().numpy().reshape(1,512,1280)
            #    data.append(record_outer)
            data = torch.tensor(data).to(device=device)
            targets = torch.tensor(targets).to(device=device)

            #test_y = np.concatenate((test_y, targets.ravel()))
            #data = data.to(device=device)
            #targets = targets.to(device=device)
            predicted_output = model(data.float())
            loss = loss_function(predicted_output, targets.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_ep_loss = np.concatenate((train_ep_loss.ravel(), loss.to(device='cpu').detach().numpy().ravel()))
        test_ep_loss, test_y, pred_y= test()[0]
            
        data = np.array(train_data['data'][index: ])
        targets = np.array(train_data['targets'][index: ])
        data = torch.tensor(data).to(device=device)
        targets = torch.tensor(targets).to(device=device)
        predicted_output = model(data.float())
        loss = loss_function(predicted_output, targets.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_ep_loss = np.concatenate((train_ep_loss.ravel(), loss.to(device='cpu').detach().numpy().ravel()))
        #test_ep_loss,  = test()[0]
            
        train_ep_loss = np.mean(train_ep_loss.ravel())
        train_loss = np.concatenate((train_loss.ravel(), train_ep_loss.ravel()))
        #test_loss = np.concatenate((test_loss.ravel(), test_ep_loss.ravel()))
        print(f'Epoch {epoch + 1} completed with Loss {train_ep_loss}')
        r, _pv = stats.pearsonr(test_y,pred_y)
        print(f'Validation completed with Loss {ep_loss}')
        print(f'Validation r: {r}')
        print()
    #print(train_loss.shape, test_loss.shape)
    torch.save(model, 'network_cnn_ep80.pth')
    torch.save(model.state_dict(), 'network_params_cnn_ep80.pth')
    print("Training Completed!")
    return train_loss, test_loss


# ---------------------


# -----------------Output
train_loss, test_loss = train()

l, test_y, pred_y = test()

#print(train_loss, test_loss)
def plot_loss():
    line1 = plt.scatter(np.arange(train_loss.ravel().shape[0]),train_loss.ravel(), c='red')
    line2 = plt.scatter(np.arange(test_loss.ravel().shape[0]),test_loss.ravel(), c='blue')
    plt.legend([line1, line2], ["train", "test"], loc=1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('test_cnn_ep80_loss.png', dpi=150)
    
def plot_test():
    plt.scatter(test_y.ravel()-5,pred_y.ravel()-5, c='blue')
    #plt.legend([line1, line2], ["train", "test"], loc=1)
    plt.xlabel('true pkd')
    plt.ylabel('pred pkd')
    plt.savefig('test_cnn_ep80.png', dpi=150)
    

plot_loss()
plot_test()
