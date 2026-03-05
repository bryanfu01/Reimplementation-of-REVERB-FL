import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cnn_model import CNN_Model
from poison_attacks import fgsm_attack, pgd_attack, awgn_attack

class Client():

    """
    Stores:
    - Personal CNN model
    - DataLoader object containing the client's data

    Functions:
    - train() runs a single iteration of forward and backprop to return the weight gradients
    """
    def __init__(self, client_data, device = None, attack_type = None):
        self.model = CNN_Model().to(device)
        self.device = device
        self.attack_type = attack_type
        self.tensor_data = client_data

    def train(self, global_weights, tau_steps = 10, learning_rate = 1E-4, wd = 1E-4, device = None):

        self.model.load_state_dict(global_weights)

        # This isn't a call to the function it is in, just think of it as a safety measure.
        self.model.train()

        # Creating optimizer object and setting up parameters
        optimizer = optim.Adam(self.model.parameters(), lr = learning_rate, weight_decay = wd)

        # To individually move through each data entry, easiest to make DataLoader
        # data object and iterator
        client_data_iter = iter(self.tensor_data)

        for step in range(tau_steps):
            try:
                # Grabs the next data entry
                data, label = next(client_data_iter)
            except StopIteration:
                # This is if reach end of dataset, reshuffles it and starts over.
                client_data_iter = iter(self.tensor_data)
                data, label = next(client_data_iter)

            data, label = data.to(device), label.to(device)

            if self.attack_type == "fgsm":
                data = fgsm_attack(self.model, data, label, device=self.device)
            elif self.attack_type == "pgd":
                data = pgd_attack(self.model, data, label, device=self.device)
            elif self.attack_type == "awgn":
                data = awgn_attack(data, device=self.device)
            
            optimizer.zero_grad()

            scores = self.model(data)
            
            # This is the paper version of poisoning
            if self.attack_type in ["fgsm", "pgd"]:
                # Create a target where the model is forced to guess randomly (10% for every class)
                uniform_targets = torch.ones_like(scores) / scores.size(1)
                
                # PyTorch cross_entropy natively accepts soft probability targets!
                loss = F.cross_entropy(scores, uniform_targets)
            else:
                loss = F.cross_entropy(scores, label)
            """
            # This is the nuclear option of poisoning
            loss = F.cross_entropy(scores, label)

            if self.attack_type in ["fgsm", "pgd"]:
                loss = -1.0 * loss
            """

            loss.backward()
            
            if self.attack_type in ["fgsm", "pgd"]:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

        # Don't need to send the change in weights, just need the new update since 
        # the old weights are constant across all the clients.
        return self.model.state_dict()
