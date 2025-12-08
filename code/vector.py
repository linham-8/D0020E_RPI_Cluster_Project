import os
import sys
import torch
import torch.distributed as dist

os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'

current_rank = int(sys.argv[1])

store = dist.FileStore("/scratch/D0020E_RPI_Cluster_Project/vector_temp", world_size=2)

dist.init_process_group(backend='gloo', store=store, rank=current_rank, world_size=2)

if current_rank == 0:
    data = torch.tensor([6.0, 7.0])
    dist.send(tensor=data, dst=1)
    print(f"Rank 0 Sent: {data}")

elif current_rank == 1:
    data = torch.zeros(2)
    dist.recv(tensor=data, src=0)
    print(f"Rank 1 Received: {data}")
    os.remove("/scratch/D0020E_RPI_Cluster_Project/vector_temp")

dist.destroy_process_group() 
