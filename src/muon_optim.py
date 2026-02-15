"""
Muon Optimizer — MomentUm Orthogonalized by Newton-schulz

Adapted from:
  - https://github.com/KellerJordan/Muon
  - reference/nanuGPT/nanugpt/optimizers/muon_optim.py

Single-GPU variant only (no DDP). Combines Muon for 2D hidden weight matrices
with AdamW for embeddings, head, and scalar params.

Usage:
    from muon_optim import configure_muon_optimizer
    optimizer = configure_muon_optimizer(model)
"""

import torch


# ---- Newton-Schulz Orthogonalization ----


def zeropower_via_newtonschulz5(G, steps=5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Uses a quintic iteration whose coefficients maximize the slope at zero.

    This does NOT produce exact UV^T but rather US'V^T where S' is diagonal with
    entries ~ Uniform(0.5, 1.5), which empirically doesn't hurt model performance.

    Operates in bfloat16 for GPU efficiency.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


# ---- Update functions ----


def muon_update(grad, momentum_buffer, beta=0.95, ns_steps=5, nesterov=True):
    """Compute Muon update: momentum + Newton-Schulz orthogonalization."""
    momentum_buffer.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum_buffer, beta) if nesterov else momentum_buffer
    if update.ndim == 4:  # conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    """Standard Adam update with bias correction."""
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


# ---- Combined Optimizer ----


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Single-GPU optimizer combining Muon (for 2D hidden matrices) with AdamW
    (for embeddings, head, scalars).

    Each param_group must have `use_muon=True/False`.

    Muon groups support momentum warmup via min_momentum/max_momentum/momentum_warmup.
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("min_momentum", None)
                group.setdefault("max_momentum", None)
                group.setdefault("momentum_warmup", None)
                group.setdefault("weight_decay", 0)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                # Momentum warmup
                if (
                    group["min_momentum"] is not None
                    and group["max_momentum"] is not None
                    and group["momentum_warmup"] is not None
                ):
                    params = group["params"]
                    if len(params) > 0:
                        step = self.state[params[0]].get("step", 0)
                        frac = min(step / max(group["momentum_warmup"], 1), 1)
                        group["momentum"] = (1 - frac) * group[
                            "min_momentum"
                        ] + frac * group["max_momentum"]

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = muon_update(
                        p.grad.float(), state["momentum_buffer"], beta=group["momentum"]
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad.float(),
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


# ---- Helper: configure optimizer from model ----


def configure_muon_optimizer(
    model,
    # Muon params (for 2D hidden matrices inside Block layers)
    muon_lr=0.05,
    muon_momentum=0.95,
    muon_momentum_min=0.85,
    muon_momentum_max=0.95,
    muon_momentum_warmup=300,
    muon_weight_decay=0.0,
    # Adam params
    head_lr=0.008,
    embed_lr=0.6,
    scalar_lr=0.04,
    adam_betas=(0.8, 0.95),
    adam_eps=1e-8,
    adam_weight_decay=0.0,
    # Structure
    block_class_name="Block",
    head_name="lm_head",
):
    """
    Partition model parameters into Muon and Adam groups.

    - Head (lm_head): AdamW
    - Embeddings (nn.Embedding): AdamW
    - Hidden 2D matrices inside Block layers: Muon
    - Scalars (<2D): AdamW
    """
    assigned = set()

    # Head params
    head_module = getattr(model, head_name)
    head_weight = head_module.weight
    head_params = [head_weight]
    assigned.add(id(head_weight))

    # Embedding params (excluding tied weights)
    embed_params = []
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            for p in module.parameters(recurse=False):
                if id(p) not in assigned:
                    embed_params.append(p)
                    assigned.add(id(p))

    # Hidden matrix params (2D+ inside Block layers)
    hidden_matrix_params = []
    for module in model.modules():
        if module.__class__.__name__ == block_class_name:
            for p in module.parameters(recurse=True):
                if p.ndim >= 2 and id(p) not in assigned:
                    hidden_matrix_params.append(p)
                    assigned.add(id(p))

    # Scalar params (anything remaining with ndim < 2)
    scalar_params = []
    for p in model.parameters():
        if p.ndim < 2 and id(p) not in assigned:
            scalar_params.append(p)
            assigned.add(id(p))

    # Build param groups
    adam_groups = []
    if head_params:
        adam_groups.append(
            dict(params=head_params, lr=head_lr, weight_decay=adam_weight_decay)
        )
    if embed_params:
        adam_groups.append(
            dict(params=embed_params, lr=embed_lr, weight_decay=adam_weight_decay)
        )
    if scalar_params:
        adam_groups.append(
            dict(params=scalar_params, lr=scalar_lr, weight_decay=adam_weight_decay)
        )
    adam_groups = [
        dict(**g, betas=adam_betas, eps=adam_eps, use_muon=False) for g in adam_groups
    ]

    muon_group = dict(
        params=hidden_matrix_params,
        lr=muon_lr,
        momentum=muon_momentum_max,
        min_momentum=muon_momentum_min,
        max_momentum=muon_momentum_max,
        momentum_warmup=muon_momentum_warmup,
        weight_decay=muon_weight_decay,
        use_muon=True,
    )

    param_groups = [*adam_groups, muon_group]

    # Print summary
    def _n(params):
        return sum(p.numel() for p in params)

    print(f"Muon optimizer groups:")
    print(
        f"  Head:    {len(head_params)} tensors, {_n(head_params):,} params (Adam, lr={head_lr})"
    )
    print(
        f"  Embed:   {len(embed_params)} tensors, {_n(embed_params):,} params (Adam, lr={embed_lr})"
    )
    print(
        f"  Scalar:  {len(scalar_params)} tensors, {_n(scalar_params):,} params (Adam, lr={scalar_lr})"
    )
    print(
        f"  Hidden:  {len(hidden_matrix_params)} tensors, {_n(hidden_matrix_params):,} params (Muon, lr={muon_lr})"
    )

    return MuonWithAuxAdam(param_groups)
