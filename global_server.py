import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cnn_model import CNN_Model

class Global_Server():
    """
    Stores:
    - Global CNN Model
    - DataLoader object containing clean 5% reserve set.

    Functions:
    - aggregate: Averages the updates of all the client weights and sets global server with new weights.
        Params
        - list of client weight updates
        Returns
        - Nothing
    - retrain: Produces new upate sent to all the servers
        Params
        - Adam optimizer parameters
        Returns
        - Nothing
    - get_weights(): Getter function returning the current weights of the model
        Params
        - None
        Returns
        - current global weights
    - compute_acc: Computes the current accuracy of the model
        Params
        - reference to full clean data set
        Returns
        - accuracy of specific round
    - reset_state: Resets the state of the global server after a simulation has finished.
    """

    def __init__(self, reserve_set, test_loader, device = None):
        self.model = CNN_Model().to(device)
        self.reserve = reserve_set
        self.device = device

        print("Pre-loading testing dataset into VRAM...")
        test_x_list = []
        test_y_list = []
        
        # We loop through the hard drive exactly ONCE here
        for x, y in test_loader:
            test_x_list.append(x)
            test_y_list.append(y)
            
        # Concatenate everything into one giant tensor and permanently bolt to GPU
        self.test_x = torch.cat(test_x_list).to(self.device, non_blocking=True)
        self.test_y = torch.cat(test_y_list).to(self.device, non_blocking=True)
        print("Testing dataset successfully locked into GPU memory.")

    def aggregate(self, client_weight_list):
        avg_weights = {}

        layer_names = client_weight_list[0].keys()

        # Iterates through the layers of the model dictionaries and averages them independently
        for layer in layer_names:
            # Vectorized technique to quickly combine all the client weights of a specific layer
            stacked_tensors = torch.stack([client[layer] for client in client_weight_list])

            avg_weights[layer] = torch.mean(stacked_tensors.float(), dim=0)

        self.model.load_state_dict(avg_weights)

    def get_weights(self):
        return self.model.state_dict()
    
    def retrain(self, num_epochs = 1, learning_rate = 1E-4, wd = 1E-4, device = None):
         # Creating optimizer object and setting up parameters
        optimizer = optim.Adam(self.model.parameters(), lr = learning_rate, weight_decay = wd)
        
        for epoch in range(num_epochs):
            for data, label in self.reserve:

                data, label = data.to(device), label.to(device)

                optimizer.zero_grad()

                scores = self.model(data)

                loss = F.cross_entropy(scores, label)

                loss.backward()

                optimizer.step()

    def compute_acc(self):
        self.model.eval()

        with torch.no_grad():
            # The GPU processes the entire dataset in a single matrix multiplication
            scores = self.model(self.test_x)
            _, preds = scores.max(1)
            
            # The comparison is perfectly vectorized
            num_correct = (preds == self.test_y).sum().item()
            num_samples = self.test_y.size(0)

        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
        
        self.model.train()

        return acc
    
    def reset_weights(self, device=None):
        self.model = CNN_Model().to(device)
