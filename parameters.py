import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N = 1_000_000
M = 100
T = 1_000
C_max = 2_000
end_full_random = 50_000
