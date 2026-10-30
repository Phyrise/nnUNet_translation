from torch.optim.lr_scheduler import _LRScheduler


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1) #removed last (..., False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


class WarmupPolyLRScheduler(_LRScheduler):
    """
    Linear warm-up, then poly decay to max_steps (one step per epoch in nnU-Net):
        step < warmup_steps:  lr = initial_lr * (step + 1) / warmup_steps
        otherwise:            lr = initial_lr * (1 - (step - warmup_steps) / (max_steps - warmup_steps)) ** exponent
    Used by the ISBI 2027 recipes: initial_lr 1e-2 with 5 warm-up epochs, or 1e-3 with 15 warm-up epochs when the
    encoder starts from nnFoundation. warmup_steps = 0 gives the PolyLRScheduler curve.
    """

    def __init__(self, optimizer, initial_lr: float, max_steps: int, warmup_steps: int, exponent: float = 0.9,
                 current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        if current_step < self.warmup_steps:
            new_lr = self.initial_lr * (current_step + 1) / self.warmup_steps
        else:
            decay_steps = max(1, self.max_steps - self.warmup_steps)
            new_lr = self.initial_lr * max(0., 1 - (current_step - self.warmup_steps) / decay_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
