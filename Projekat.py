import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_covtype
from sklearn.metrics import precision_recall_fscore_support, classification_report


# Ovo su parovi X - y gdje X predstavlja vrijednost nekih parametara a y predstavlja rezultat (tip sumskog pokrivaca)
data = fetch_covtype()

# Varijabla za parametre
X = data['data']

# Varijabla za rezultantne vrijednosti
y = data['target'] - 1


# Standardizacijom dobijam podatke manjih vrijednosti i blizi su jedni drugima
# Koriste se neke matematicke operacije, dijele se podaci sa srednjih vrijednostima itd.
scaler = StandardScaler()
X = scaler.fit_transform(X)


unique_targets = np.unique(data['target'])
num_unique_targets = len(unique_targets)
print("Number of unique target values:", num_unique_targets)


# Ima 7 mogucih izlaznih vrijednosti vrijednosti za predvidjanje dakle broj izlaznih neurona ce biti 7 - int
# 54 parametra na ulazu  - float i 580 000 parova podataka


# Kreiramo klasicnu, potpuno povezanu mrezu (feedforward)

# Podaci podijeljeni na trening, test i validacione  ---  25% test i onda od ostatka 25% je za validaciju, ostalo je za trening
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.20, random_state=1)


# Konverzija iz Numpy array u Pytorch tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)


# Sad imam:

# X_train - trening vrijednosti parametara
# X_test -  testne vrijednosti parametara
# X_val -   validacione vrijednosti parametara

# y_train - trening rezultati
# y_test -  testni rezultati - treba da se racunaju 
# y_val -   validacioni rezultati

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, neurons_per_layer):
        super(Model, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, neurons_per_layer))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(neurons_per_layer, output_size))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
# Hiperparametri

OUTPUT_SIZE = 7
INPUT_SIZE = 54
hidden_layers = 2
neurons_per_layer = 60
learning_rate = 0.05
num_epochs = 20
batch_size = 60
patience = 3
loss_function = nn.CrossEntropyLoss()
model = Model(INPUT_SIZE, OUTPUT_SIZE, hidden_layers, neurons_per_layer)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


minimum_validation_loss = float('inf')
counter = 0

# TRENIRANJE MREZE
model.train()       # Trening mod

for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        batch_x = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        outputs = model(batch_x)                                    # Pretpostavke
        loss = loss_function(outputs, batch_y)                      # Mjeri razliku izmedju pretpostavljenih i tacnih vrijednosti
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()                                            # Koriguje parametre modela


    model.eval()        # Mod za validaciju
    with torch.no_grad():
        val_outputs = model(X_val)                                  # Pretpostavke
        val_loss = loss_function(val_outputs, y_val)                # Mjeri razliku izmedju pretpostavljenih i tacnih vrijednosti



    print(f'Epoch {epoch+1}, Training loss: {loss.item()}, Validation loss: {val_loss.item()}')

    
    
    if val_loss < minimum_validation_loss:
        minimum_validation_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping on epoch", epoch + 1)
            break


print("\nNeural network done training!\n")

# TESTIRANJE MREZE
model.eval()  # Mod za validaciju


with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)

print(predicted)

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, predicted, average='macro')
print(f'Precision: {precision}, Recall: {recall}, F1-score: {f1_score}')
