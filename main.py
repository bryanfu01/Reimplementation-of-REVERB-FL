import torch
from simulator import Simulator

def main():
    num_clients = 10
    simulation = Simulator(num_clients=num_clients)
    num_rounds = 60
    """
    simulation.run_simulation(
        attack_type="fgsm", 
        attack_ratio=0.5, 
        num_rounds=60, 
        checkpoint_path="/content/drive/MyDrive/reverb_fl_checkpoint_round_5.pt"
    )
    """
    simulation.run_simulation(attack_type = None, framework_active=False, num_rounds=num_rounds)
    simulation.run_simulation(attack_type = 'fgsm', framework_active=False, num_rounds=num_rounds)
    simulation.run_simulation(attack_type = 'pgd', framework_active=False, num_rounds=num_rounds)
    simulation.run_simulation(attack_type = 'awgn', framework_active=False, num_rounds=num_rounds)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)

    main()