import torch
from simulator import Simulator

def main():
    num_clients = 10
    simulation = Simulator(num_clients=num_clients)
    num_rounds = 60
    simulation.run_simulation(num_rounds=num_rounds)
    simulation.run_simulation(attack_type = 'fgsm', num_rounds=num_rounds)
    simulation.run_simulation(attack_type = 'pgd', num_rounds=num_rounds)
    simulation.run_simulation(attack_type = 'awgn', num_rounds=num_rounds)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)

    main()