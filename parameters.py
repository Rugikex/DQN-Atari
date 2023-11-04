import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_update = 50_000
replay_memory_maxlen = 600_000
update_target_network = 10_000
seconds_per_training = 3600 * 9
steps_per_episode = 2_000
epsilon_final_step = 1_000_000
