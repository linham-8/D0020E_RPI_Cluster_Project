import os 
import sys 
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'

current_rank = int(sys.argv[1])

store=dist.FileStore("/scratch/D0020E_RPI_Cluster_Project/svm_shared_file", world_size=4)

dist.init_process_group(backend='gloo', store=store, rank=current_rank, world_size=4)

def load(path, offset):
    with open(path, 'rb') as f:
        return torch.frombuffer(bytearray(f.read()[offset:]), dtype=torch.uint8)

X = load('/scratch/mnist_dataset/emnist-digits-train-images-idx3-ubyte', 16).float().reshape(-1, 784) / 255.0
Y = load('/scratch/mnist_dataset/emnist-digits-train-labels-idx1-ubyte', 8).long()
X = X[current_rank::4]
Y = Y[current_rank::4]

model = DDP(torch.nn.Linear(784, 10))
opt = torch.optim.SGD(model.parameters(), lr=0.1)
crit = torch.nn.MultiMarginLoss()

if current_rank == 0:
    for epoch in range(5):
        for i in range(0, len(X), 64):
            opt.zero_grad()
            loss = crit(model(X[i:i+64]), Y[i:i+64])
            loss.backward()
            opt.step()

if current_rank == 0:
    torch.save(model.state_dict(), "/scratch/D0020E_RPI_Cluster_Project/final_model.pt")
    Xt = load('/scratch/mnist_dataset/emnist-digits-test-images-idx3-ubyte', 16).float().reshape(-1, 784) / 255.0
    Yt = load('/scratch/mnist_dataset/emnist-digits-test-labels-idx1-ubyte', 8).long()
    
    acc = (model.module(Xt).argmax(dim=1) == Yt).float().mean() * 100
    print(f"Accuracy: {acc:.2f}%")
    os.remove("/scratch/D0020E_RPI_Cluster_Project/svm_shared_file")

dist.destroy_process_group()
