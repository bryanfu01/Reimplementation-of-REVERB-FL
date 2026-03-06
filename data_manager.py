import torch
import numpy as np
from torch.utils.data import TensorDataset, Subset, DataLoader
from data_conversion import SpeechCommandsComplexSTFT

class DataManager:
    """
    Manages allocation of data across global server and clients.
    Stores the monolithic clean dataset and partitions indices for IID distribution.
    """

    def __init__(self, num_clients = 10, is_iid = True, data_root = "./data"):
        self.training_data = SpeechCommandsComplexSTFT(root = data_root, subset = "training")
        self.testing_data = SpeechCommandsComplexSTFT(root = data_root, subset = "testing")
        self.num_clients = num_clients
        
        # Call the partitioning function immediately upon initialization
        if is_iid:
            self.client_partitions = self._iid_partition()
        else:
            self.client_partitions = self._non_iid_partition()

    def _iid_partition(self):
        client_dict = {}
        num_samples = len(self.training_data) # Use len() to get the size of the dataset
        
        # 1. Generate a completely randomized list of all possible indices
        shuffled_indices = torch.randperm(num_samples).numpy()
        
        # 2. Slice off the exact 5% for the Server's Reserve Set
        num_reserve = int(num_samples * 0.05)
        client_dict["reserve_set"] = shuffled_indices[:num_reserve].tolist()
        
        # 3. The remaining 85% goes into a pool for the clients
        client_pool = shuffled_indices[num_reserve:]
        
        client_chunks = np.array_split(client_pool, self.num_clients)

        for i, chunk in enumerate(client_chunks):
            client_dict[i] = chunk.tolist()
            
        return client_dict
    
    def _non_iid_partition(self, alpha=0.5):
        client_dict = {}
        num_samples = len(self.training_data)
        
        # 1. Generate a completely randomized list of all possible indices
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        # 2. Slice off the exact 5% for the Server's Reserve Set (Kept balanced/IID)
        num_reserve = int(num_samples * 0.05)
        reserve_indices = indices[:num_reserve]
        client_dict["reserve_set"] = reserve_indices.tolist()
        
        # 3. The remaining 95% goes into a pool for the clients
        client_pool_indices = indices[num_reserve:]
        
        # Extract labels to know how to split the classes
        # (If your dataset has .targets or .labels, use that for speed. Otherwise, we iterate once)
        if hasattr(self.training_data, 'targets'):
            targets = np.array(self.training_data.targets)
        else:
            print("Extracting labels for Non-IID split (this happens once)...")
            targets = np.array([label for _, label in self.training_data])
            
        pool_targets = targets[client_pool_indices]
        
        # 4. Group the client pool indices by their true class label
        num_classes = len(np.unique(targets))
        class_indices = {c: [] for c in range(num_classes)}
        
        for i, true_idx in enumerate(client_pool_indices):
            label = pool_targets[i]
            class_indices[label].append(true_idx)
            
        # 5. Initialize empty lists for each client
        for i in range(self.num_clients):
            client_dict[i] = []
            
        # 6. Distribute each class using the Dirichlet distribution
        for c in range(num_classes):
            c_idx = class_indices[c]
            np.random.shuffle(c_idx) # Shuffle indices within the class
            
            # The magic Dirichlet math
            proportions = np.random.dirichlet(np.repeat(alpha, self.num_clients))
            splits = (proportions * len(c_idx)).astype(int)
            
            current_idx = 0
            for i in range(self.num_clients):
                chunk_size = splits[i]
                
                # If it's the last client, give them all remaining indices (to catch rounding errors)
                if i == self.num_clients - 1:
                    assigned = c_idx[current_idx:]
                else:
                    assigned = c_idx[current_idx : current_idx + chunk_size]
                    
                client_dict[i].extend(assigned)
                current_idx += chunk_size
                
        # 7. Shuffle each client's assigned list so data isn't perfectly grouped by class
        for i in range(self.num_clients):
            np.random.shuffle(client_dict[i])
            client_dict[i] = list(client_dict[i]) # Ensure it's a standard Python list
            
        return client_dict

    def get_client_loader(self, client_id, batch_size=16):
        """
        Returns a DataLoader holding the reference to the clean data and the specific indices belonging to a client.
        """
        indices = self.client_partitions[client_id]
        
        # Subset creates the "pointer + indices" relationship seamlessly
        subset = torch.utils.data.Subset(self.training_data, indices)
        
        # DataLoader handles the batching automatically
        return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    def get_reserve_loader(self, batch_size = 32):
        """
        Returns a DataLoader holding reference to clean data and indices for reserve set.
        """

        indices = self.client_partitions["reserve_set"]

         # Subset creates the "pointer + indices" relationship seamlessly
        subset = torch.utils.data.Subset(self.training_data, indices)
        
        # DataLoader handles the batching automatically
        return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    def get_test_loader(self, batch_size = 1024):
        return DataLoader(self.testing_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
