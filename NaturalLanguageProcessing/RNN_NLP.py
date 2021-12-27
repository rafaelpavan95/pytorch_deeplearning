import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

class NLP(nn.Module):
  def __init__(self, n_vocab, embed_dim, n_hidden, n_rnnlayers, n_outputs, device):
    super(NLP, self).__init__()
    self.V = n_vocab
    self.D = embed_dim
    self.M = n_hidden
    self.K = n_outputs
    self.L = n_rnnlayers
    self.device = device

    self.embed = nn.Embedding(self.V, self.D)
    self.rnn = nn.LSTM(
        input_size=self.D,
        hidden_size=self.M,
        num_layers=self.L,
        batch_first=True)

    self.fc1 = nn.Linear(self.M, n_outputs)

  def forward(self, X):
    # initial hidden states
    h0 = torch.zeros(self.L, X.size(0), self.M).to(self.device)
    c0 = torch.zeros(self.L, X.size(0), self.M).to(self.device)
    # embedding layer
    # turns word indexes into word vectors
    out = self.embed(X)
    # get RNN unit output
    out, _ = self.rnn(out, (h0, c0))
    out, _ = torch.max(out, 1)
    # we only want h(T) at the final time step
    out = self.fc1(out)

    return out

def training_nn(model, criterion, optimizer, train_loader, test_loader, epochs, early_stop, device, weight):
  train_losses = []
  test_losses = []

  flag = 0

  for it in range(epochs):
    train_loss = []
    for inputs, targets in train_loader:
      # print("inputs.shape:", inputs.shape, "targets.shape:", targets.shape)
      targets = targets.view(-1, 1).float()
      # move data to GPU
      inputs, targets = inputs.to(device), targets.to(device)

      # zero the parameter gradients
      optimizer.zero_grad()

      # Forward pass
      outputs = model(inputs)
      y = targets
      weight_ = weight[y.data.view(-1).long()].view_as(y)
  
      loss = criterion(outputs, y)
      loss = loss*weight_
      loss = loss.mean()
      # Backward and optimize
      loss.backward()
      optimizer.step()

      train_loss.append(loss.item())

    # Get train loss and test loss
    train_loss = np.mean(train_loss) # a little misleading
    
    test_loss = []
    for inputs, targets in test_loader:
      inputs, targets = inputs.to(device), targets.to(device)
      targets = targets.view(-1, 1).float()
      outputs = model(inputs)
      y = targets
      weight_ = weight[y.data.view(-1).long()].view_as(y)
      loss = criterion(outputs, y)
      loss = loss*weight_
      loss = loss.mean()
      
      test_loss.append(loss.item())
    test_loss = np.mean(test_loss)

    # Save losses
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    if it>1:
        
      if np.min(np.asarray(test_losses))<test_losses[it]:

        flag = flag + 1

      else: flag = 0

    
    if flag >early_stop:

      print('End of the Algorithm. The Neural Network didnt improve in the last 2 epochs')
      break # stop if overfitting
      



    print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, \
      Test Loss: {test_loss:.4f}, Flag: {flag}')
  
  return train_losses, test_losses, model


def plot_learning_curves(train_losses, test_losses, title='Learning Curve'):

    plt.plot(train_losses, label='Train Loss', color='red')
    plt.plot(test_losses, label='Test Loss', color='blue')
    plt.legend()
    plt.title(title)
    plt.grid()
    plt.show()


def accuracy( model, train_loader,test_loader):
# Accuracy

  p_train = []
  y_train = []
  for inputs, targets in train_loader:
    targets = targets.view(-1, 1).float()

    # Forward pass
    outputs = model(inputs)

    # Get prediction
    predictions = list((outputs > 0).cpu().numpy())
    
    # Store predictions
    p_train += predictions
    y_train += list(targets.cpu().numpy())

  p_train = np.array(p_train)
  y_train = np.array(y_train)
  train_acc = np.mean(y_train == p_train)


  p_test = []
  y_test = []
  for inputs, targets in test_loader:
    targets = targets.view(-1, 1).float()

    # Forward pass
    outputs = model(inputs)

    # Get prediction
    predictions = list((outputs > 0).cpu().numpy())
    
    # Store predictions
    p_test += predictions
    y_test += list(targets.cpu().numpy())

  p_test = np.array(p_test)
  y_test = np.array(y_test)
  test_acc = np.mean(y_test == p_test)
  print(f"Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")

  return y_test, p_test


def plot_confusion_matrix_(cm, classes,title='Confusion matrix',cmap=plt.cm.Blues):

    print('Confusion matrix')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


