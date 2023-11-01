import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N = 2_000
M = 666
C_max = 500
frame_per_trainings = 16_000
