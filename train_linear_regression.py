import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader


class ToyDataset(Dataset):
    def __init__(self, length: int = 8):
        super().__init__()
        self.length = length

    def __len__(self):
        return 8

    def __getitem__(self, idx: int):
        return torch.tensor([idx], dtype=torch.float32), torch.tensor(
            [idx * 2 + 1], dtype=torch.float32
        )


def setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, device_id=rank
    )


def cleanup():
    dist.destroy_process_group()


def main(rank: int, world_size: int):
    print("rank: ", rank, "world_size: ", world_size)

    setup(rank, world_size)

    # prepare dataset and dataloader
    batch_size = 2
    num_batches = 2

    dataset = ToyDataset(length=world_size * batch_size * num_batches)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)

    # prepare model
    model = nn.Linear(1, 1, bias=True).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-1)

    for i in range(100):
        for X, Y in loader:
            X, Y = X.to(rank), Y.to(rank)
            pred = ddp_model(X)
            loss = F.mse_loss(pred, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.detach()

            # in place operation
            dist.all_reduce(loss)
            global_loss = loss / dist.get_world_size()

            if dist.get_rank() == 0:
                print(f"[rank {rank}] loss={global_loss.cpu().item()}")

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
