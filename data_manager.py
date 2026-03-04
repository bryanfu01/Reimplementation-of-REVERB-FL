import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from data_conversion import SpeechCommandsComplexSTFT

class DataManager:
    """
    Manages allocation of data across global server and clients.
    Stores the monolithic clean dataset and partitions indices for IID distribution.
    """

    def __init__(self, num_clients = 10, data_root = "./data"):
        self.training_data = SpeechCommandsComplexSTFT(root = data_root, subset = "training")
        self.testing_data = SpeechCommandsComplexSTFT(root = data_root, subset = "testing")
        self.num_clients = num_clients
        
        # Call the partitioning function immediately upon initialization
        self.client_partitions = self._iid_partition()

    def _iid_partition(self):
        client_dict = {}
        num_samples = len(self.training_data) # Use len() to get the size of the dataset
        
        # 1. Generate a completely randomized list of all possible indices
        # If num_samples is 100, this creates a shuffled array like [45, 2, 99, 14...]
        shuffled_indices = torch.randperm(num_samples).numpy()
        
        # 2. Slice off the exact 5% for the Server's Reserve Set
        num_reserve = int(num_samples * 0.05)
        client_dict["reserve_set"] = shuffled_indices[:num_reserve].tolist()
        
        # 3. The remaining 85% goes into a pool for the clients
        client_pool = shuffled_indices[num_reserve:]
        
        # 4. Use numpy.array_split to evenly divide the pool
        # This brilliantly handles edge cases where the math doesn't divide perfectly!
        client_chunks = np.array_split(client_pool, self.num_clients)
        
        # 5. Assign chunks to the dictionary
        for i, chunk in enumerate(client_chunks):
            client_dict[i] = chunk.tolist()
            
        return client_dict

    def get_client_loader(self, client_id, batch_size=16):
        """
        Returns a DataLoader holding the reference to the clean data and the specific indices belonging to a client.
        """
        indices = self.client_partitions[client_id]
        
        # Subset creates the "pointer + indices" relationship seamlessly
        subset = torch.utils.data.Subset(self.training_data, indices)
        
        # DataLoader handles the batching automatically
        return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    def get_reserve_loader(self, batch_size = 32):
        """
        Returns a DataLoader holding reference to clean data and indices for reserve set.
        """

        indices = self.client_partitions["reserve_set"]

         # Subset creates the "pointer + indices" relationship seamlessly
        subset = torch.utils.data.Subset(self.training_data, indices)
        
        # DataLoader handles the batching automatically
        return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    def get_test_loader(self, batch_size = 100):
        return DataLoader(self.testing_data, batch_size=batch_size, shuffle=False, num_workers=2)
