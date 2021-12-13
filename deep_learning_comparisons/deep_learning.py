import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image

from google.colab import drive
drive.mount('/content/drive')

######################################################################
# OneLayerNetwork
######################################################################

class OneLayerNetwork(torch.nn.Module):
    def __init__(self):
        super(OneLayerNetwork, self).__init__()
        self.linear = torch.nn.Linear(784, 3)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

######################################################################
# TwoLayerNetwork
######################################################################

class TwoLayerNetwork(torch.nn.Module):
    def __init__(self):
        super(TwoLayerNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(784, 400)
        self.linear2 = torch.nn.Linear(400, 3)

    def forward(self, x):
        layer1 = torch.sigmoid(self.linear1(x))
        outputs = self.linear2(layer1)
        return outputs

# load data from csv
def load_data(filename):
    data = np.loadtxt(filename)
    y = data[:, 0].astype(int)
    X = data[:, 1:].astype(np.float32) / 255
    return X, y

# plot one example
def plot_img(x):
    x = x.reshape(28, 28)
    img = Image.fromarray(x*255)
    plt.figure()
    plt.imshow(img)
    return

def evaluate_loss(model, criterion, dataloader):
    model.eval()
    total_loss = 0.0
    for batch_X, batch_y in dataloader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate_acc(model, dataloader):
    model.eval()
    total_acc = 0.0
    for batch_X, batch_y in dataloader:
        outputs = model(batch_X)
        predictions = torch.argmax(outputs, dim=1)
        total_acc += (predictions==batch_y).sum()
        
    return total_acc / len(dataloader.dataset)

def train(model, criterion, optimizer, train_loader, valid_loader):
    train_loss_list = []
    valid_loss_list = []
    train_acc_list = []
    valid_acc_list = []
    for epoch in range(1, 31):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
        train_loss = evaluate_loss(model, criterion, train_loader)
        valid_loss = evaluate_loss(model, criterion, valid_loader)
        train_acc = evaluate_acc(model, train_loader)
        valid_acc = evaluate_acc(model, valid_loader)
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)

        print(f"| epoch {epoch:2d} | train loss {train_loss:.6f} | train acc {train_acc:.6f} | valid loss {valid_loss:.6f} | valid acc {valid_acc:.6f} |")

    return train_loss_list, valid_loss_list, train_acc_list, valid_acc_list

######################################################################
# main
######################################################################

def main():

    # fix random seed
    np.random.seed(0)
    torch.manual_seed(0)

    # load data with correct file path
    data_directory_path =  "/content/drive/My Drive/ComSciM146/data"


    X_train, y_train = load_data(os.path.join(data_directory_path, "hw3_train.csv"))
    X_valid, y_valid = load_data(os.path.join(data_directory_path, "hw3_valid.csv"))
    X_test, y_test = load_data(os.path.join(data_directory_path, "hw3_test.csv"))


    # print out three training images with different labels
    n, d = X_train.shape
    used_labels = []
    for i in range(n):
      if y_train[i] in used_labels:
        continue
      plot_img(X_train[i])
      used_labels.append(y_train[i])
      if len(used_labels) >= 3:
        break

    print("Data preparation...")

    # convert numpy arrays to tensors
    t_X_train, t_y_train = torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long)
    t_X_valid, t_y_valid = torch.Tensor(X_valid), torch.tensor(y_valid, dtype=torch.long)
    t_X_test, t_y_test = torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long)



    # prepare dataloaders for training, validation, and testing
    train_loader = DataLoader(TensorDataset(t_X_train, t_y_train), batch_size=10)
    valid_loader = DataLoader(TensorDataset(t_X_valid, t_y_valid), batch_size=10)
    test_loader = DataLoader(TensorDataset(t_X_test, t_y_test), batch_size=10)
    

 
    ### part e: prepare OneLayerNetwork, criterion, and optimizer
    model_one = OneLayerNetwork()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_one.parameters(), lr = 0.0005)

    print("Start training OneLayerNetwork...")
    results_one = train(model_one, criterion, optimizer, train_loader, valid_loader)
    print("Done!")


    # prepare TwoLayerNetwork, criterion, and optimizer
    model_two = TwoLayerNetwork()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_two.parameters(), lr=0.0005)

    print("Start training TwoLayerNetwork...")
    results_two = train(model_two, criterion, optimizer, train_loader, valid_loader)
    print("Done!")

    one_train_loss, one_valid_loss, one_train_acc, one_valid_acc = results_one
    two_train_loss, two_valid_loss, two_train_acc, two_valid_acc = results_two


    # generate a plot to comare one_train_loss, one_valid_loss, two_train_loss, two_valid_loss
    plt.show()
    plt.plot(one_train_loss, 'red', label='One Train Loss')
    plt.plot(one_valid_loss, 'orange', label='One Valid Loss')
    plt.plot(two_train_loss, 'green', label='Two Train Loss')
    plt.plot(two_valid_loss, 'blue', label='Two Valid Loss')
    plt.legend()
    plt.title("SGD Loss")
    plt.show()



    # generate a plot to comare one_train_acc, one_valid_acc, two_train_acc, two_valid_acc
    plt.plot(one_train_acc, 'red', label='One Train Acc')
    plt.plot(one_valid_acc, 'orange', label='One Valid Acc')
    plt.plot(two_train_acc, 'green', label='Two Train Acc')
    plt.plot(two_valid_acc, 'blue', label='Two Valid Acc')
    plt.legend()
    plt.title("SGD Accuracy")
    plt.show()



    # calculate the test accuracy
    one_test_acc = evaluate_acc(model_one, test_loader)
    two_test_acc = evaluate_acc(model_two, test_loader)
    print("OneLayerNetwork test accuracy: ", one_test_acc)
    print("TwoLayerNetwork test accuracy: ", two_test_acc)


    # replace the SGD optimizer with the Adam optimizer and do the experiments again
    model_one = OneLayerNetwork()
    optimizer = torch.optim.Adam(model_one.parameters())
    results_one = train(model_one, criterion, optimizer, train_loader, valid_loader)

    model_two= TwoLayerNetwork()
    optimizer = torch.optim.Adam(model_two.parameters())
    results_two = train(model_two, criterion, optimizer, train_loader, valid_loader)

    one_train_loss, one_valid_loss, one_train_acc, one_valid_acc = results_one
    two_train_loss, two_valid_loss, two_train_acc, two_valid_acc = results_two

    plt.plot(one_train_loss, 'red', label='One Train Loss')
    plt.plot(one_valid_loss, 'orange', label='One Valid Loss')
    plt.plot(two_train_loss, 'green', label='Two Train Loss')
    plt.plot(two_valid_loss, 'blue', label='Two Valid Loss')
    plt.legend()
    plt.title("Adam Loss")
    plt.show()

    plt.plot(one_train_acc, 'red', label='One Train Acc')
    plt.plot(one_valid_acc, 'orange', label='One Valid Acc')
    plt.plot(two_train_acc, 'green', label='Two Train Acc')
    plt.plot(two_valid_acc, 'blue', label='Two Valid Acc')
    plt.legend()
    plt.title("Adam Accuracy")
    plt.show()

    one_test_acc = evaluate_acc(model_one, test_loader)
    two_test_acc = evaluate_acc(model_two, test_loader)
    print("OneLayerNetwork Adam test accuracy: ", one_test_acc)
    print("TwoLayerNetwork Adam test accuracy: ", two_test_acc)




if __name__ == "__main__":
    main()