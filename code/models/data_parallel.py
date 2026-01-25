import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json

os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
rank = int(sys.argv[1])
world_size = 5

print(f"Rank {rank}: Trying to connect")

dist.init_process_group(backend='gloo', rank=rank, world_size=world_size, 
store=dist.FileStore("/scratch/temp/data_parallel_sync", world_size=world_size))

print(f"Rank {rank}: Connected")


def load(path, offset):
    with open(path, 'rb') as f:
        return torch.frombuffer(bytearray(f.read()[offset:]), dtype=torch.uint8)

X = load('/scratch/mnist_dataset/emnist-digits-train-images-idx3-ubyte', 16).float().reshape(-1, 784) / 255.0
Y = load('/scratch/mnist_dataset/emnist-digits-train-labels-idx1-ubyte', 8).long()
X = X[rank::world_size]
Y = Y[rank::world_size]

model = DDP(torch.nn.Linear(784, 10))
opt = torch.optim.SGD(model.parameters(), lr=0.01)
crit = torch.nn.CrossEntropyLoss()

for epoch in range(5):
    for i in range(0, len(X), 64):
        opt.zero_grad()
        loss = crit(model(X[i:i+64]), Y[i:i+64])
        loss.backward()
        opt.step()

if rank == 0:
    torch.save(model.state_dict(), "/scratch/temp/data_parallel_model.pt")

    Xt = load('/scratch/mnist_dataset/emnist-digits-test-images-idx3-ubyte', 16).float().reshape(-1, 784) / 255.0
    Yt = load('/scratch/mnist_dataset/emnist-digits-test-labels-idx1-ubyte', 8).long()

    with torch.no_grad():
        acc = (model(Xt).argmax(dim=1) == Yt).float().mean() * 100

    print(f"Accuracy: {acc:.2f}%")
    log = {"accuracy": float(acc)}
    with open("/scratch/temp/data_parallel.log", "w") as f:
        f.write(json.dumps(log))

dist.destroy_process_group()