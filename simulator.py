import torch
import random
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from client import Client
from data_manager import DataManager
from global_server import Global_Server

class Simulator():
    def __init__(self, num_clients = 10, is_iid = True, path = "./speech_command_dataset"):
        self.current_round = 0
        self.client_list = list(range(num_clients))
        self.acc_history = []
        self.malicious_client_list = None
        self.attack_type = None
        self.framework_active = False

        self.is_iid = is_iid

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )

        # Instantiating DataManager object
        self.data_manager = DataManager(num_clients=num_clients, is_iid=is_iid, data_root=path)

        # Slicing off reserve set for server
        reserve_data = self.data_manager.get_reserve_loader()
        # Slicing off testing set for server
        testing_set = self.data_manager.get_test_loader()
        # Creating server and giving reserve set
        self.global_server = Global_Server(reserve_set=reserve_data, test_loader=testing_set, device = self.device)

    def run_simulation(self, attack_type = None, framework_active = False, attack_ratio = 0.5, num_rounds = 60, pretrain_rounds = 3, checkpoint_path = None):

        initial_lr = 1e-4
        decay_rate = 0.9
        decay_steps = 1000.0

        if checkpoint_path is None:
            print(f"\n--- Initializing new run. Attack: {attack_type} ---")
            self.acc_history = []
            self.current_round = 0
            self.global_server.reset_weights(device=self.device)
            self.framework_active = framework_active
            self.attack_type = attack_type
            num_attackers = int(attack_ratio * len(self.client_list))
            self.malicious_client_list = set(random.sample(self.client_list, num_attackers))

            # First need to pretrain server on reserve set if framework active
            if self.framework_active:
                print("Pre-training server on reserve set...")
                self.global_server.retrain(num_epochs=pretrain_rounds, learning_rate=initial_lr, attack_type=self.attack_type, device = self.device)
        else:
            print(f"\n--- Resuming from checkpoint: {checkpoint_path} ---")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # 1. Restore the mathematical timeline
            self.current_round = checkpoint['round']
            self.acc_history = checkpoint['acc_history']
            self.attack_type = checkpoint['attack_type']
            self.framework_active = checkpoint['framework_active']
            self.data_manager.client_partitions = checkpoint['client_partitions']
            self.malicious_client_list = checkpoint['malicious_client_list']
            
            # 2. Inject the saved weights directly into the server's model
            self.global_server.model.load_state_dict(checkpoint['model_state_dict'])

            # Need global server to retain the same 5% reserve set
            self.global_server.reserve = self.data_manager.get_reserve_loader()
            
            print(f"Successfully loaded. Resuming at round {self.current_round}.")

        total_client_steps = self.current_round * 10

        for i in range(self.current_round, num_rounds):
            start_time = time.perf_counter()
            current_lr = initial_lr * (decay_rate ** (total_client_steps / decay_steps))

            acc = self._run_round(current_lr)
            self.acc_history.append(acc)
            self.current_round += 1
            total_client_steps += 10
            end_time = time.perf_counter()
            print('Communication round time taken: %d s' % (end_time - start_time))

            if (i + 1) % 5 == 0:
                checkpoint_data = {
                    'round': i + 1,
                    'model_state_dict': self.global_server.get_weights(),
                    'acc_history': self.acc_history,
                    'attack_type': self.attack_type,
                    'framework_active': self.framework_active,
                    'client_partitions': self.data_manager.client_partitions,
                    'malicious_client_list': self.malicious_client_list
                }

                if self.attack_type is None:
                    attack_label = "No Attack"
                else:
                    attack_label = self.attack_type.upper()

                if self.framework_active:
                    framework_type = "REVERB-FL"
                else:
                    framework_type = "Baseline_FedAvg"
                if self.is_iid:
                    distribution = "iid"
                else:
                    distribution = "non-iid"

                file_name = f'/content/drive/MyDrive/{framework_type}_{attack_label}_{distribution}_checkpoint_round_{i+1}_only_reserve.pt'

                torch.save(checkpoint_data, file_name)
                print(f"Checkpoint saved for Round {i+1} of {attack_label}!")
        
        self._plot_acc()

    def _run_round(self, current_lr):
        # Measures total client training time
        client_start_time = time.perf_counter()
        weight_update_list = self._train_clients(current_lr)
        client_end_time = time.perf_counter()
        print('Total client training time taken: %.2f s' % (client_end_time - client_start_time))
        
        self.global_server.aggregate(weight_update_list)

        if self.framework_active:
            server_start_time = time.perf_counter()
            self.global_server.retrain(learning_rate=current_lr, attack_type=self.attack_type, device = self.device)
            server_end_time = time.perf_counter()
            print('Total server retraining time taken: %.2f s' % (server_end_time - server_start_time))

        return self.global_server.compute_acc()


    def _train_clients(self, current_lr):
        client_weight_list = []
        for client_id in self.client_list:
            # Gets client data and stores uses it to initiate new client object
            client_data = self.data_manager.get_client_loader(client_id)
            if client_id in self.malicious_client_list:
                client = Client(client_data, device=self.device, attack_type=self.attack_type)
            else:
                client = Client(client_data, device = self.device)

            # Gets current global weights and instantiates weights of client while training
            global_weights = self.global_server.get_weights()

            weight_update = client.train(global_weights=global_weights, learning_rate=current_lr, device = self.device)
            # Adds to list of client weights
            client_weight_list.append(weight_update)
        return client_weight_list 

    def _plot_acc(self):
        """
        Takes a list of accuracy decimals (e.g. [0.10, 0.45, 0.65]) 
        and plots them as a line graph over the communication rounds.
        """
        # 1. Generate the x-axis (Round numbers: 1, 2, 3, ...)
        rounds = range(1, len(self.acc_history) + 1)
        
        # 2. Convert decimal accuracies to percentages for a cleaner Y-axis
        percentages = [acc * 100 for acc in self.acc_history]

        # 3. Create the figure
        plt.figure(figsize=(10, 6))
        
        # 4. Plot the line with markers at each data point
        plt.plot(rounds, percentages, marker='o', linestyle='-', color='b', label='Global Model Accuracy')
        
        if self.attack_type is None:
            attack_label = "No Attack"
        else:
            attack_label = self.attack_type.upper()

        if self.framework_active:
            framework_type = "REVERB-FL"
        else:
            framework_type = "Baseline_FedAvg"
        
        if self.is_iid:
            diststribution = "iid"
        else:
            diststribution = "non-iid"
        # 5. Add labels, title, and grid
        plt.title(f'{framework_type}: Global Model Accuracy with Attack Type: {attack_label} and Distribution: {diststribution} Only Reserve')
        plt.xlabel('Communication Round')
        plt.ylabel('Test Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 6. Adjust layout and display the plot
        filename = f'{framework_type}_accuracy_{attack_label}_with_{diststribution}_distribution.png'
        
        plt.savefig(filename, bbox_inches='tight')
        print(f"Graph successfully saved as {filename}")



