import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#using GPU instead of CPU if available if available for faster training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#all the Hyper-Parameters 
input = 784 # 28x28
hidden1 = 128
hidden2 = 64
hidden3 = 64
output = 10
epochs = 20
batch_size = 64
learning_rate = 0.001

# FashionMNIST dataset 
train_dataset = torchvision.datasets.FashionMNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.FashionMNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


# Fully connected neural network with three hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input, hidden1, hidden2, hidden3, output):
        super(NeuralNet, self).__init__()
        self.z1 = nn.Linear(input, hidden1) 
        self.relu = nn.ReLU()
        self.z2 = nn.Linear(hidden1, hidden2)  
        self.z3 = nn.Linear(hidden2, hidden3)
        self.z4 = nn.Linear(hidden3, output)
    
    def forward(self, x):
        out = self.z1(x)
        out = self.relu(out)
        out = self.z2(out)
        out = self.relu(out)
        out = self.z3(out)
        out = self.relu(out)
        out = self.z4(out)
        
        return out

model = NeuralNet(input, hidden1, hidden2, hidden3, output).to(device)

def train_the_model(train_loader):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

    # Train the model
    n_train = len(train_loader)

    for epoch in range(epochs):
        total = 0
        correct = 0
        final_loss = 0.0
        model.train()

        for x_train, y_train in train_loader:  
            x_train = x_train.reshape(-1, 28*28).to(device)
            y_train = y_train.to(device)

            # Forward
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track accuracy and loss
            _, predicted = torch.max(y_pred.data, 1)
            total += y_train.size(0)
            correct += (predicted == y_train).sum().item()
            final_loss += loss.item()

        acc = 100 * correct / total
        print(f"Epoch {epoch + 1} - Loss: {final_loss:.4f}, Accuracy: {acc:.2f}%")


train_the_model(train_loader)

#getting the test accuracy
with torch.no_grad():
    correct = 0
    n_test = len(test_loader.dataset)

    for x_test, y_test in test_loader:
        x_test = x_test.reshape(-1, 28*28).to(device)
        y_test = y_test.to(device)

        y_pred_test = model(x_test)

        
        _, predicted = torch.max(y_pred_test, 1)
        correct += (predicted == y_test).sum().item()

    acc = correct / n_test
    print(f'Accuracy of the network on the {n_test} test images: {100*acc} %')