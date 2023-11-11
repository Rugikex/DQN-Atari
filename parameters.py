import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
START_UPDATE = 50_000
REPLAY_MEMORY_MAXLEN = 600_000
UPDATE_TARGET_NETWORK = 10_000
STEPS_PER_EPISODE = 2_000
EPSILON_FINAL_STEP = 1_000_000
