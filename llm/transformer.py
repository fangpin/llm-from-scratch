import torch
import math
import einx
from typing import overload, TypeAlias, Any
from collections.abc import Callable, Iterable
from torch import Tensor
from jaxtyping import Float


ParamsT: TypeAlias = Iterable[torch.Tensor] | Iterable[dict[str, Any]] | Iterable[tuple[str, torch.Tensor]]


class Linear(torch.nn.Module):
    def __init__(
        self, in_features, out_features, weights: Float[Tensor, " out in"] | None = None, device=None, dtype=None
    ):
        super().__init__()
        if weights is None:
            sigma = math.sqrt(2.0 / (in_features + out_features))
            self.w = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
            torch.nn.init.trunc_normal_(self.w, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)
        else:
            self.w = torch.nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einx.dot("... [in], out [in] -> ... out", x, self.w)


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.embeddings = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.embeddings, mean=0.0, std=1 / math.sqrt(embedding_dim))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]


class RmsNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.g = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.g * x).to(input_dtype)


class SiLu(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * x


class Glu(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(in_features, out_features, device=device, dtype=dtype)
        self.w2 = Linear(in_features, out_features, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.w1(x)) * self.w2(x)


class SwiGlu(torch.nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, device=None, dtype=None) -> None:
        super().__init__()
        self.w1 = Linear(d_in, d_hidden, device=device, dtype=dtype)
        self.w3 = Linear(d_in, d_hidden, device=device, dtype=dtype)
        self.w2 = Linear(d_hidden, d_out, device=device, dtype=dtype)
        self.silu = SiLu()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))


FFN = SwiGlu


class RoPE(torch.nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # inv_freq: (dim//2,)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))

        # t: (seq_len,)
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        # freqs: (seq_len, dim//2)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # outer product

        emb = freqs.repeat_interleave(2, dim=-1)  # (seq_len, dim)

        # Now register buffers
        self.register_buffer("cos_cached", emb.cos().to(dtype))  # (seq_len, dim)
        self.register_buffer("sin_cached", emb.sin().to(dtype))

    def forward(self, x: Float[Tensor, "... seq d_k"], token_positions: Float[Tensor, "... seq"]) -> torch.Tensor:
        # token_positions: (..., seq_len) 任意前缀维度
        # x: (..., seq_len, dim)
        cos = self.cos_cached[token_positions]  # (..., seq_len, dim)
        sin = self.sin_cached[token_positions]  # (..., seq_len, dim)

        x_reshaped = x.view(*x.shape[:-1], -1, 2)  # (..., seq_len, dim//2, 2)
        x_rotated = torch.stack((-x_reshaped[..., 1], x_reshaped[..., 0]), dim=-1)  # rotate: (a,b) -> (-b,a)
        x_rotated = x_rotated.view(*x.shape)  # (..., seq_len, dim)

        if x.ndim == 4:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        # print(f"x shape {x.shape}, cos shape {cos.shape}")
        x_rot = x * cos + x_rotated * sin
        return x_rot


class Softmax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim=-1) -> torch.Tensor:
        max_x = torch.max(x, dim, keepdim=True).values
        x = x - max_x
        x_exp = torch.exp(x)
        x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
        return x_exp / x_exp_sum


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()

    def forward(
        self,
        q: Float[Tensor, "... s d"],
        k: Float[Tensor, "... s d"],
        v: Float[Tensor, "... s d"],
        mask: torch.Tensor
        | None = None,  # true means the item should be covered and not participating to softmax calculating
    ) -> torch.Tensor:
        d_model = q.shape[-1]

        # Compute attention scores
        att = einx.dot("... s_q [d], ... s_k [d] -> ... s_q s_k", q, k)
        att_scale = att / math.sqrt(d_model)

        if mask is not None:
            if mask.ndim < att_scale.ndim:
                mask = mask.reshape((1,) * (att_scale.ndim - mask.ndim) + mask.shape)
            # Apply mask - removed the ~ operator
            att_scale = att_scale.masked_fill(mask, -1e9)

        att_score = self.softmax(att_scale)

        return einx.dot("... s_q [s], ... [s] d -> ... s_q d", att_score, v)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_head: int, max_seq_len=2048, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.out_linear = Linear(d_model, d_model, device=device, dtype=dtype)
        self.project = Linear(in_features=d_model, out_features=3 * d_model, device=device, dtype=dtype)
        self.dot_product_att = ScaledDotProductAttention()

        # Cache causal mask - removed the ~ operator
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: Float[Tensor, "b s d"]) -> torch.Tensor:
        seq_len = x.shape[1]

        mask = self.causal_mask[:seq_len, :seq_len]

        qkv = self.project(x)
        q, k, v = einx.rearrange("b s (n h d) -> n b h s d", qkv, n=3, h=self.num_head)

        output = self.dot_product_att(q, k, v, mask)
        output = einx.rearrange("b h s d -> b s (h d)", output)
        return self.out_linear(output)


class MultiHeadAttentionWithRoPE(MultiHeadAttention):
    def __init__(self, d_model: int, num_head: int, theta: float = 10000, max_seq_len=2048, device=None, dtype=None):
        super().__init__(d_model=d_model, num_head=num_head, max_seq_len=max_seq_len, device=device, dtype=dtype)
        self.rope = RoPE(d_model // num_head, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[1]
        batch_size = x.shape[0]

        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        mask = self.causal_mask[:seq_len, :seq_len]

        qkv = self.project(x)
        q, k, v = einx.rearrange("b s (n h d) -> n b h s d", qkv, n=3, h=self.num_head)

        # Apply RoPE to q and k
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        output = self.dot_product_att(q, k, v, mask)
        output = einx.rearrange("b h s d -> b s (h d)", output)
        return self.out_linear(output)


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        theta: float = 10000,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.rms_norm1 = RmsNorm(d_model, device=device, dtype=dtype)
        self.rms_norm2 = RmsNorm(d_model, device=device, dtype=dtype)
        self.mult_head_atten = MultiHeadAttentionWithRoPE(
            d_model, num_heads, theta, max_seq_len=max_seq_len, device=device, dtype=dtype
        )
        self.ffe = FFN(d_model, d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x_norm = self.rms_norm1(x)
        x_atten = self.mult_head_atten(x_norm, token_positions)
        x = x + x_atten
        x_norm = self.rms_norm2(x)
        x_ffe = self.ffe(x_norm)
        return x + x_ffe


class Transformer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        num_layers: int,
        max_seq_len=2048,
        rope_theta: float = 10000,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=max_seq_len,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RmsNorm(d_model=d_model, device=device, dtype=dtype)
        self.out_linear = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.max_seq_len = max_seq_len

    def forward(self, token_ids: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x, token_positions)
        x_norm = self.norm(x)
        logits = self.out_linear(x_norm)
        return logits


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Reshape for cross_entropy, handling any shape of logits
        # logits = einops.rearrange(logits, "... c -> (...) c")
        # targets = einops.rearrange(targets, "... -> (...)")
        logits = einx.rearrange("... c -> (...) c", logits)
        targets = einx.rearrange("... -> (...)", targets)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        correct_log_probs = log_probs[torch.arange(len(log_probs)), targets]
        nll = -correct_log_probs

        mean_loss = torch.mean(nll)

        return mean_loss


class SGDDecay(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3) -> None:
        if lr < 0:
            raise ValueError(f"invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr=1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay=1e-3,
        eps=1e-8,
    ):
        if lr < 0:
            raise ValueError(f"invalid learning rate: {lr}")
        beta1, beta2 = betas
        defaults = {
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "weight_decay": weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["sm"] = torch.zeros_like(p.data)

                m, sm = state["m"], state["sm"]
                t = state["t"] + 1

                grad = p.grad.data

                # Update biased first moment estimate
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                # Update biased second raw moment estimate
                sm.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                m_hat = m / (1.0 - beta1**t)
                sm_hat = sm / (1.0 - beta2**t)

                # Update parameters
                p.data.addcdiv_(m_hat, torch.sqrt(sm_hat) + eps, value=-lr)

                # Weight decay
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

                state["t"] = t
        return loss


def cos_lr_scheduler(it: int, warmup_iters: int, cos_cycle_iters: int, lr_min: float, lr_max: float) -> float:
    if it <= warmup_iters:
        return lr_max * it / warmup_iters
    elif warmup_iters < it < cos_cycle_iters:
        return lr_min + 0.5 * (lr_max - lr_min) * (
            1 + math.cos(math.pi * (it - warmup_iters) / (cos_cycle_iters - warmup_iters))
        )
    else:
        return lr_min


def gradient_clip(params: Iterable[torch.nn.Parameter], max_norm: float, delta=1e-6):
    with torch.no_grad():
        grads = [p.grad for p in params if p.grad is not None]
        total_norm = torch.linalg.norm(torch.stack([torch.linalg.norm(g.detach()) for g in grads]))
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + delta)
            for g in grads:
                g.detach().mul_(clip_coef)


if __name__ == "__main__":
    for lr in [1e1, 1e2, 1e3]:
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGDDecay([weights], lr=lr)
        for t in range(10):
            opt.zero_grad()  # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean()  # Compute a scalar loss value.
            print(loss.cpu().item())
            loss.backward()  # Run backward pass, which computes gradients.
            opt.step()  # Run optimizer step.
            print(f"lr={lr}, t={t}, loss={loss}")

