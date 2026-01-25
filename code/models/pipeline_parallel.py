import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import json

os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
rank = int(sys.argv[1])
world_size = 5

print(f"Rank {rank}: Trying to connect")

dist.init_process_group(backend='gloo', rank=rank, world_size=world_size,
                        store=dist.FileStore("/scratch/temp/pipeline_parallel_sync", world_size=world_size))

print(f"Rank {rank}: Connected")

def load(path, offset):
    with open(path, 'rb') as f:
        return torch.frombuffer(bytearray(f.read()[offset:]), dtype=torch.uint8)

if rank == 0:
    X = load('/scratch/mnist_dataset/emnist-digits-train-images-idx3-ubyte', 16).float().reshape(-1, 784) / 255.0
    Y = load('/scratch/mnist_dataset/emnist-digits-train-labels-idx1-ubyte', 8).long()

    Xt = load('/scratch/mnist_dataset/emnist-digits-test-images-idx3-ubyte', 16).float().reshape(-1, 784) / 255.0
    Yt = load('/scratch/mnist_dataset/emnist-digits-test-labels-idx1-ubyte', 8).long()

    dist.broadcast(torch.tensor([1]), src=0)

    for epoch in range(5):
        for i in range(0, len(X), 64):
            batch_x = X[i:i+64]
            batch_y = Y[i:i+64]
            if len(batch_x) < 64: continue

            dist.broadcast(torch.tensor([1]), src=0)
            dist.send(batch_x, dst=1)

            preds = torch.zeros(64, 10)
            dist.recv(preds, src=4)
            preds.requires_grad = True
            loss = nn.CrossEntropyLoss()(preds, batch_y)

            grad_initial = torch.autograd.grad(loss, preds)[0]
            dist.send(grad_initial, dst=4)

    dist.broadcast(torch.tensor([0]), src=0)
    dist.broadcast(torch.tensor([2]), src=0)
    
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(Xt), 64):
            batch_xt = Xt[i:i+64]
            if len(batch_xt) < 64: continue

            dist.broadcast(torch.tensor([1]), src=0)
            dist.send(batch_xt, dst=1)

            preds = torch.zeros(64, 10)
            dist.recv(preds, src=4)

            acc = (preds.argmax(dim=1) == Yt[i:i+64]).sum().item()
            correct += acc
            total += 64

    dist.broadcast(torch.tensor([0]), src=0)

    final_acc = (correct / total) * 100
    print(f"Accuracy: {final_acc:.2f}%")
    
    log = {"accuracy": float(final_acc)}
    with open("/scratch/temp/pipeline_parallel.log", "w") as f:
        f.write(json.dumps(log))

else:
    if rank == 1:
        model = nn.Sequential(nn.Linear(784, 128), nn.ReLU())
    elif rank == 4:
        model = nn.Linear(128, 10)
    else:
        model = nn.Sequential(nn.Linear(128, 128), nn.ReLU())

    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    mode_signal = torch.tensor([0])
    dist.broadcast(mode_signal, src=0)

    if mode_signal.item() == 1:
        step_signal = torch.tensor([0])
        while True:
            dist.broadcast(step_signal, src=0)
            if step_signal.item() == 0: break

            if rank == 1:
                input_data = torch.zeros(64, 784)
                dist.recv(input_data, src=0)
            else:
                input_data = torch.zeros(64, 128)
                dist.recv(input_data, src=rank-1)

            input_data.requires_grad = True
            output = model(input_data)

            if rank == 4:
                dist.send(output, dst=0)
            else:
                dist.send(output, dst=rank+1)

            if rank == 4:
                grad_in = torch.zeros(64, 10)
                dist.recv(grad_in, src=0)
            else:
                grad_in = torch.zeros(64, 128)
                dist.recv(grad_in, src=rank+1)

            opt.zero_grad()
            output.backward(grad_in)
            opt.step()

            if rank > 1:
                dist.send(input_data.grad, dst=rank-1)

    torch.save(model.state_dict(), f"/scratch/temp/pipeline_parallel_model_rank{rank}.pt")

    dist.broadcast(mode_signal, src=0)
    if mode_signal.item() == 2:
        step_signal = torch.tensor([0])
        with torch.no_grad():
            while True:
                dist.broadcast(step_signal, src=0)
                if step_signal.item() == 0: break

                if rank == 1:
                    input_data = torch.zeros(64, 784)
                    dist.recv(input_data, src=0)
                else:
                    input_data = torch.zeros(64, 128)
                    dist.recv(input_data, src=rank-1)

                output = model(input_data)

                if rank == 4:
                    dist.send(output, dst=0)
                else:
                    dist.send(output, dst=rank+1)

dist.destroy_process_group()