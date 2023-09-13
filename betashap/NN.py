from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    
    def __init__(self, x, y):
        
        """
        TODO: Store `seqs`. to `self.x` 
        
        Note that you DO NOT need to covert them to tensor as we will do this later.
        Do NOT permute the data.
        """
        
        # your code here
        self.x = x.astype('float32')
        self.y = y.astype('float32')
    
    def __len__(self):
        
        """
        TODO: Return the number of samples (i.e. patients).
        """
        
        # your code here
        return len(self.x)
    
    def __getitem__(self, index):
        
        """
        TODO: Generates one sample of data.
        """
        
        # your code here
        return self.x[index], self.y[index]

class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TabularMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        out = self.encoder(x)
        return out
    
    def loss_function(self, y_hat, y):
        # Reconstruction loss
        BCE = nn.functional.binary_cross_entropy(y_hat.squeeze(), y, reduction='sum')
        return BCE
    
    def fit(self, x_train, y_train, batch_size=128, learning_rate=1e-3, num_epochs=100):
        # Create a DataLoader for the input data
        dataset = CustomDataset(x_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Define the optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Train the model
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(dataloader):
                # Get the input data
                inputs, y = data
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                y_hat = self(inputs)
                loss = self.loss_function(y_hat, y)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            # Print the average loss for the epoch
            print('Epoch %d - Loss: %.3f' % (epoch + 1, running_loss / len(dataloader)))
        #save the model
        torch.save(self.state_dict(), 'nn_model.pt')
        
    def evaluate(self, x_test, y_test):
        # Create a DataLoader for the input data
        dataset = CustomDataset(x_test, y_test)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)
        
        y_hats =[]
        
        # Evaluate the model
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                # Get the input data
                inputs,y = data
                
                # Forward pass
                y_hat = self(inputs)
                y_hats.extend(y_hat)
        auc = roc_auc_score(y_test, y_hats)
        print("length of auc list: ", len(y_hats))
        print( "AUC: ", auc)
    