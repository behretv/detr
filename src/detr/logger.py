# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Logging utilities: metric tracking and git-SHA reporting."""

from __future__ import annotations

import datetime
import os
import subprocess
import time
from collections import defaultdict, deque

import torch
from tqdm import tqdm


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if not isinstance(v, (float, int)):
                raise TypeError(
                    f"Expected float or int, got {type(v).__name__} for key '{k}'"
                )
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter}")
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        MB = 1024.0 * 1024.0
        with tqdm(
            total=len(iterable), desc=header, dynamic_ncols=True, leave=True
        ) as pbar:
            for i, obj in enumerate(iterable):
                data_time.update(time.time() - end)
                yield obj
                iter_time.update(time.time() - end)
                if i % print_freq == 0 or i == len(iterable) - 1:
                    postfix = str(self)
                    if torch.cuda.is_available():
                        mem = torch.cuda.max_memory_allocated() / MB
                        postfix += self.delimiter + f"max mem: {mem:.0f}"
                    postfix += self.delimiter + f"time: {iter_time}"
                    postfix += self.delimiter + f"data: {data_time}"
                    pbar.set_postfix_str(postfix)
                pbar.update(1)
                end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        tqdm.write(
            f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)"
        )


def get_sha() -> str:
    """Return a string with the current git SHA, diff status, and branch."""
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    return f"sha: {sha}, status: {diff}, branch: {branch}"
