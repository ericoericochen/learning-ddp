import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchvision.datasets import MNIST


def setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, device_id=rank
    )
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def main(rank: int, world_size: int):
    print("rank: ", rank, "world_size: ", world_size)

    setup(rank, world_size)

    # download dataset using process on rank 0
    if rank == 0:
        print("downloading mnist dataset from rank: ", rank)
        dataset = MNIST(root="/tmp/mnist", download=True, train=True)

    dist.barrier()

    # load dataset on rest of ranks
    if rank != 0:
        print("loading dataset on rank: ", rank)
        dataset = dataset = MNIST(root="/tmp/mnist", download=False, train=True)

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
