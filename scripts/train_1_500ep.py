import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# -------------------Load Data
train_df = pd.read_pickle("../data/train_data.pkl")
test_df = pd.read_pickle("../data/test_data.pkl")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# concatenate mol_embed, protein_embed, and pKd + 5
train_data = []
for index, row in train_df.iterrows():
    train_data.append(np.concatenate((row['mol embed'],
                                      row['protein embed'],
                                      np.array([np.log(row['Kd (nM)']) + 5]))))
train_data = np.array(train_data)

# 
test_data = []
for index, row in test_df.iterrows():
    test_data.append(np.concatenate((row['mol embed'],
                                     row['protein embed'],
                                     np.array([np.log(row['Kd (nM)']) + 5]))))
test_data = np.array(test_data)


# ---------------------------------


# -------------------Define network
#1792 -(elu)-> 2480 -(dropout 0.15, elu)-> 1280 -(dropout 0.15, elu)-> 512 -(dropout 0.15, elu)-> 256 -(dropout 0.15, elu)-> 
class NN(nn.Module):
    def __init__(self, num_features):
        super(NN, self).__init__()
        self.num_features = num_features
        self.layer1 = nn.Sequential(nn.Linear(num_features, 2480), nn.ELU())
        self.layer2 = nn.Sequential(nn.Linear(2480, 1280), nn.Dropout(p=0.15), nn.ELU())
        self.layer3 = nn.Sequential(nn.Linear(1280, 512), nn.Dropout(p=0.15), nn.ELU())
        self.layer4 = nn.Sequential(nn.Linear(512, 256), nn.Dropout(p=0.15), nn.ELU())
        self.layer5 = nn.Sequential(nn.Linear(256, 128), nn.Dropout(p=0.15), nn.ELU())
        self.layer6 = nn.Sequential(nn.Linear(128, 64), nn.ELU())
        self.ouput_layer = nn.Linear(64, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.ouput_layer(x)
        return x


num_epochs = 500
batch_size = 512
learning_rate = 0.001
num_features = 1792

model = NN(num_features=num_features)
loss_function = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device=device)
model.train()


# ---------------------------------

# ----------------Train and Test func
def test():
    # model = torch.load('network_1_ep50.pth')
    model.eval()

    test_y = np.array([])
    pred_y = np.array([])

    ep_loss = np.array([])
    for batch_idx, batch_data in enumerate(test_loader):
        data, targets = (torch.split(batch_data, [1792, 1], dim=1))
        test_y = np.concatenate((test_y, targets.detach().numpy().ravel()))
        data = data.to(device=device)
        targets = targets.to(device=device)


        predicted_output = model(data.float())
        pred_y = np.concatenate((pred_y, predicted_output.to(device='cpu').detach().numpy().ravel()))
        loss = loss_function(predicted_output, targets.float())
        ep_loss = np.concatenate((ep_loss.ravel(), loss.to(device='cpu').detach().numpy().ravel()))

    print(f'Validation completed with Loss {loss}')
    return np.mean(ep_loss.ravel()), test_y, pred_y


def train():
    train_loss = np.array([])
    test_loss = np.array([])
    for epoch in range(num_epochs):
        train_ep_loss = np.array([])
        # train_ep_loss = np.array([])
        for batch_idx, batch_data in enumerate(train_loader):
            # data = batch_data
            # targets = train_y_loader[batch_idx]
            data, targets = (torch.split(batch_data, [1792, 1], dim=1))

            data = data.to(device=device)
            targets = targets.to(device=device)

            predicted_output = model(data.float())
            loss = loss_function(predicted_output, targets.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_ep_loss = np.concatenate((train_ep_loss.ravel(), loss.to(device='cpu').detach().numpy().ravel()))

        test_ep_loss = test()[0]
        train_ep_loss = np.mean(train_ep_loss.ravel())
        train_loss = np.concatenate((train_loss.ravel(), train_ep_loss.ravel()))
        test_loss = np.concatenate((test_loss.ravel(), test_ep_loss.ravel()))
        print(f'Epoch {epoch + 1} completed with Loss {train_ep_loss}')
        print()
    print(train_loss.shape, test_loss.shape)
    torch.save(model, 'network_1_ep500.pth')
    torch.save(model.state_dict(), 'network_params_1_ep500.pth')
    print("Training Completed!")
    return train_loss, test_loss


# ---------------------


# -----------------Output
train_loss, test_loss = train()

#print(train_loss, test_loss)
line1 = plt.scatter(np.arange(train_loss.ravel().shape[0]),train_loss.ravel(), c='red')
line1 = plt.scatter(np.arange(test_loss.ravel().shape[0]),test_loss.ravel(), c='blue')
plt.legend([line1, line2], ["train", "test"], loc=1)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('test_1_ep500_loss.png', dpi=150)
# --------------------------------

