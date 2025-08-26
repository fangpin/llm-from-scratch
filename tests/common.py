from __future__ import annotations

import pathlib
from functools import lru_cache


import os

import torch
import torch.distributed as dist
import torch.nn as nn

FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent) / "fixtures"


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


def validate_ddp_net_equivalence(net):
    # Helper to validate synchronization of nets across ranks.
    net_module_states = list(net.module.state_dict().values())
    # Check that all tensors in module's state_dict() are equal.
    for t in net_module_states:
        tensor_list = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, t)
        for tensor in tensor_list:
            assert torch.allclose(tensor, t)


class _FC2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 50, bias=True)
        self.fc.bias.requires_grad = False

    def forward(self, x):
        x = self.fc(x)
        return x


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10, bias=False)
        self.fc2 = _FC2()
        self.fc3 = nn.Linear(50, 5, bias=False)
        self.relu = nn.ReLU()
        self.no_grad_fixed_param = nn.Parameter(torch.tensor([2.0, 2.0]), requires_grad=False)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ToyModelWithTiedWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=False)
        self.fc3 = nn.Linear(50, 10, bias=False)
        self.fc4 = nn.Linear(10, 50, bias=False)
        self.fc5 = nn.Linear(50, 5, bias=False)
        self.fc4.weight = self.fc2.weight
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def _setup_process_group(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"
    # https://discuss.pytorch.org/t/should-local-rank-be-equal-to-torch-cuda-current-device/150873/2
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        local_rank = None
        if device_count > 0:
            local_rank = rank % device_count
            torch.cuda.set_device(local_rank)
        else:
            raise ValueError("Unable to find CUDA devices.")
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def _cleanup_process_group():
    # Synchronize before we destroy the process group
    dist.barrier()
    dist.destroy_process_group()
