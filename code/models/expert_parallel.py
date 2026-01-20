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

def load(path, offset):
    with open(path, 'rb') as f:
        return torch.frombuffer(bytearray(f.read()[offset:]), dtype=torch.uint8)

if rank == 0:
    X = load('/scratch/mnist_dataset/emnist-digits-train-images-idx3-ubyte', 16).float().reshape(-1, 784) / 255.0
    Y = load('/scratch/mnist_dataset/emnist-digits-train-labels-idx1-ubyte', 8).long()

    Xt = load('/scratch/mnist_dataset/emnist-digits-test-images-idx3-ubyte', 16).float().reshape(-1, 784) / 255.0
    Yt = load('/scratch/mnist_dataset/emnist-digits-test-labels-idx1-ubyte', 8).long()

    gate_model = nn.Linear(784, 4)
    gate_opt = torch.optim.SGD(gate_model.parameters(), lr=0.01)

    dist.broadcast(torch.tensor([1]), src=0)

    for epoch in range(5):
        perm = torch.randperm(len(X))
        X_shuff = X[perm]
        Y_shuff = Y[perm]

        for i in range(0, len(X), 64):
            batch_x = X_shuff[i:i+64]
            batch_y = Y_shuff[i:i+64]
            if len(batch_x) < 64: continue

            dist.broadcast(torch.tensor([1]), src=0)
            
            with torch.no_grad():
                gate_scores = gate_model(batch_x)
                expert_assignments = torch.argmax(gate_scores, dim=1) + 1

            expert_losses = []
            for r in range(1, 5):
                mask = (expert_assignments == r)
                sub_x = batch_x[mask]
                sub_y = batch_y[mask]
                
                count = torch.tensor([len(sub_x)])
                dist.send(count, dst=r)
                
                if len(sub_x) > 0:
                    dist.send(sub_x, dst=r)
                    pred_buffer = torch.zeros(len(sub_x), 10)
                    dist.recv(pred_buffer, src=r)
                    
                    pred_buffer.requires_grad = True
                    loss = nn.CrossEntropyLoss()(pred_buffer, sub_y)
                    loss.backward()
                    expert_losses.append(loss.item())
                    dist.send(pred_buffer.grad, dst=r)

            gate_opt.zero_grad()
            gate_loss = torch.tensor(expert_losses).sum()
            gate_loss.requires_grad = True
            gate_opt.step()

    dist.broadcast(torch.tensor([0]), src=0)
    dist.broadcast(torch.tensor([2]), src=0)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(Xt), 64):
            batch_xt = Xt[i:i+64]
            batch_yt = Yt[i:i+64]
            if len(batch_xt) < 64: continue

            dist.broadcast(torch.tensor([1]), src=0)
            gate_scores = gate_model(batch_xt)
            expert_assignments = torch.argmax(gate_scores, dim=1) + 1
            batch_preds = torch.zeros(64, 10)

            for r in range(1, 5):
                mask = (expert_assignments == r)
                sub_x = batch_xt[mask]
                
                count = torch.tensor([len(sub_x)])
                dist.send(count, dst=r)
                
                if len(sub_x) > 0:
                    dist.send(sub_x, dst=r)
                    pred_buffer = torch.zeros(len(sub_x), 10)
                    dist.recv(pred_buffer, src=r)
                    batch_preds[mask] = pred_buffer
            
            acc = (batch_preds.argmax(dim=1) == batch_yt).sum().item()
            correct += acc
            total += 64

    dist.broadcast(torch.tensor([0]), src=0)
    print(f"Accuracy: {(correct/total)*100:.2f}%")

else:
    model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    mode_signal = torch.tensor([0])
    dist.broadcast(mode_signal, src=0)

    if mode_signal.item() == 1:
        step_signal = torch.tensor([0])
        while True:
            dist.broadcast(step_signal, src=0)
            if step_signal.item() == 0: break

            count_tensor = torch.tensor([0])
            dist.recv(count_tensor, src=0)
            curr_count = count_tensor.item()
            
            if curr_count > 0:
                input_data = torch.zeros(curr_count, 784)
                dist.recv(input_data, src=0)
                
                input_data.requires_grad = True
                opt.zero_grad()
                output = model(input_data)
                
                dist.send(output, dst=0)
                
                grad_in = torch.zeros(curr_count, 10)
                dist.recv(grad_in, src=0)
                
                output.backward(grad_in)
                opt.step()

    dist.broadcast(mode_signal, src=0)
    if mode_signal.item() == 2:
        step_signal = torch.tensor([0])
        with torch.no_grad():
            while True:
                dist.broadcast(step_signal, src=0)
                if step_signal.item() == 0: break

                count_tensor = torch.tensor([0])
                dist.recv(count_tensor, src=0)
                curr_count = count_tensor.item()
                
                if curr_count > 0:
                    input_data = torch.zeros(curr_count, 784)
                    dist.recv(input_data, src=0)
                    output = model(input_data)
                    dist.send(output, dst=0)

dist.destroy_process_group()
