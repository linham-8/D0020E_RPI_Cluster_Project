import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn

os.environ['GLOO_SOCKET_IFNAME'] = 'eth0' 
rank = int(sys.argv[1])
world_size = 5

dist.init_process_group(backend='gloo', rank=rank, world_size=world_size,
                        store=dist.FileStore("/scratch/temp/svm_shared_file", world_size=world_size))
compute_group = dist.new_group([1, 2, 3, 4])

def load(path, offset):
    with open(path, 'rb') as f:
        return torch.frombuffer(bytearray(f.read()[offset:]), dtype=torch.uint8)

if rank == 0:
    X = load('/scratch/mnist_dataset/emnist-digits-train-images-idx3-ubyte', 16).float().reshape(-1, 784) / 255.0
    Y = load('/scratch/mnist_dataset/emnist-digits-train-labels-idx1-ubyte', 8).long()
    X, Y = X[Y < 8], Y[Y < 8]

    Xt = load('/scratch/mnist_dataset/emnist-digits-test-images-idx3-ubyte', 16).float().reshape(-1, 784) / 255.0
    Yt = load('/scratch/mnist_dataset/emnist-digits-test-labels-idx1-ubyte', 8).long()
    Xt, Yt = Xt[Yt < 8], Yt[Yt < 8]

    dist.broadcast(torch.tensor([1]), src=0) 
    
    for epoch in range(5):
        for i in range(0, len(X), 64):
            batch_x = X[i:i+64]
            batch_y = Y[i:i+64]
            if len(batch_x) < 64: continue

            dist.broadcast(torch.tensor([1]), src=0)
            dist.broadcast(batch_x, src=0)
            dist.broadcast(batch_y, src=0)

    dist.broadcast(torch.tensor([0]), src=0)

    dist.broadcast(torch.tensor([2]), src=0)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(0, len(Xt), 64):
            batch_xt = Xt[i:i+64]
            if len(batch_xt) < 64: continue

            dist.broadcast(torch.tensor([1]), src=0)
            dist.broadcast(batch_xt, src=0)
            
            preds = []
            for r in range(1, 5):
                buffer = torch.zeros(64, 2)
                dist.recv(buffer, src=r)
                preds.append(buffer)
            
            full_logits = torch.cat(preds, dim=1)
            acc = (full_logits.argmax(dim=1) == Yt[i:i+64]).sum().item()
            correct += acc
            total += 64
            
    dist.broadcast(torch.tensor([0]), src=0)
    print(f"Accuracy: {((correct/total) * 100):.2f}%")

else:
    model = nn.Linear(784, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    mode_signal = torch.tensor([0])
    
    dist.broadcast(mode_signal, src=0)
    if mode_signal.item() == 1:
        batch_x = torch.zeros(64, 784)
        batch_y = torch.zeros(64, dtype=torch.long)
        step_signal = torch.tensor([0])

        while True:
            dist.broadcast(step_signal, src=0)
            if step_signal.item() == 0: break

            dist.broadcast(batch_x, src=0)
            dist.broadcast(batch_y, src=0)

            opt.zero_grad()
            local_out = model(batch_x)

            gathered = [torch.zeros(64, 2) for _ in range(4)]
            dist.all_gather(gathered, local_out.data, group=compute_group)
            
            full_parts = []
            for r, tensor in enumerate(gathered):
                if r == (rank - 1):
                    full_parts.append(local_out)
                else:
                    full_parts.append(tensor)
            
            full_logits = torch.cat(full_parts, dim=1)
            loss = nn.CrossEntropyLoss()(full_logits, batch_y)
            loss.backward()
            opt.step()

    dist.broadcast(mode_signal, src=0)
    if mode_signal.item() == 2:
        batch_x = torch.zeros(64, 784)
        step_signal = torch.tensor([0])

        with torch.no_grad():
            while True:
                dist.broadcast(step_signal, src=0)
                if step_signal.item() == 0: break

                dist.broadcast(batch_x, src=0)
                local_out = model(batch_x)
                dist.send(local_out, dst=0)

dist.destroy_process_group()
