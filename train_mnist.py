import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam


def setup(rank: int, world_size: int):
    # address of master node
    os.environ["MASTER_ADDR"] = "localhost"

    # port master listens on for communications from workers
    os.environ["MASTER_PORT"] = "12355"

    # get the accelerator - cuda, mps, mtia, xpu
    acc = torch.accelerator.current_accelerator()
    # nccl for cuda
    backend = torch.distributed.get_default_backend_for_device(acc)

    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # every process uses a different GPU in a multi-GPU setup
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor):
        return self.model(x)


# receives rank (0 <= rank < world_size)
def main(
    rank: int,
    world_size: int,
):
    setup(rank, world_size)
    print("running ddp on rank: ", rank)

    model = Model().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = Adam(ddp_model.parameters(), lr=3e-4)

    X = torch.ones(2, 8).to(rank)
    Y = torch.ones(2, 1).to(rank)

    for i in range(10):
        out = ddp_model(X)
        loss = F.mse_loss(out, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"{i} | loss={loss.cpu().item()}")

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
