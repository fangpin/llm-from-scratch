import torch
import math
import einx
from typing import Optional, overload, Union, TypeAlias, Dict, Tuple, Any
from collections.abc import Callable, Iterable


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        sigma = math.sqrt(2.0 / (in_features + out_features))
        self.w = torch.nn.Parameter(
            torch.empty(in_features, out_features, device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(
            self.w, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einx.dot("b... [in], [in] out -> b... out", x, self.w)


class Embedding(torch.nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, device=None, dtype=None
    ):
        super().__init__()
        self.embeddings = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(
            self.embeddings, mean=0.0, std=1 / math.sqrt(embedding_dim)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return einx.get_at("v d, b s -> b s d", self.embeddings, token_ids)


class RmsNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.g = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_model
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = einx.mean("... [c] -> ... 1", x**2)
        x_norm = x / torch.sqrt(rms + self.eps)
        ret = x_norm * self.g
        return ret.to(in_dtype)


class SiLu(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * x


class SwiGlu(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(in_features, out_features, device=device, dtype=dtype)
        self.w2 = Linear(in_features, out_features, device=device, dtype=dtype)
        self.silu = SiLu()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.silu(self.w1(x)) * self.w2(x)


class FFN(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.glu = SwiGlu(in_features, hidden_features, device=device, dtype=dtype)
        self.w_out = Linear(hidden_features, out_features, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(self.glu(x))


class RoPE(torch.nn.Module):
    def __init__(
        self, dim: int, max_seq_len: int = 2048, theta=10000, device=None, dtype=None
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Pre-compute the rotary embeddings in float32 for precision
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        )
        t = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos().to(dtype))
        self.register_buffer("sin_cached", emb.sin().to(dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (batch, seq_len, num_heads, head_dim) or (batch, seq_len, dim)
        # where head_dim or dim is self.dim

        seq_len = x.shape[1]

        # Get the rotary embeddings for the given sequence length
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        # Reshape for broadcasting, handling both 3D and 4D tensors
        if x.ndim == 4:
            cos = einx.rearrange("s d -> 1 s 1 d", cos)
            sin = einx.rearrange("s d -> 1 s 1 d", sin)
        elif x.ndim == 3:
            cos = einx.rearrange("s d -> 1 s d", cos)
            sin = einx.rearrange("s d -> 1 s d", sin)
        else:
            raise ValueError("Input tensor must be 3D or 4D")

        # Rotate half of the dimensions using einx
        x1, x2 = einx.rearrange("... (d2 d) -> d2 ... d", x, d2=2)
        rotated_x = torch.cat((-x2, x1), dim=-1)

        x_rot = x * cos + rotated_x * sin
        return x_rot


class Softmax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_x = einx.max("... [c] -> ... 1", x)
        x = x - max_x
        x_exp = torch.exp(x)
        x_exp_sum = einx.sum("... [c] -> ... 1", x_exp)
        return x_exp / x_exp_sum


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        d_model = q.shape[-1]

        att = einx.dot("... s d, ... t d -> ... s t", q, k)
        att_scale = att / math.sqrt(d_model)

        if mask is not None:
            assert mask.ndim >= 2
            while mask.ndim < att_scale.ndim:
                mask = mask.unsqueeze(0)
            att_scale = att_scale.masked_fill(mask, -1e9)

        att_score = self.softmax(att_scale)
        return att_score @ v


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self, d_model: int, num_head: int, max_seq_len=2048, device=None, dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.out_linear = Linear(d_model, d_model, device=device, dtype=dtype)
        self.project = Linear(
            in_features=d_model, out_features=3 * d_model, device=device, dtype=dtype
        )
        self.dot_product_att = ScaledDotProductAttention()
        self.rope = RoPE(
            d_model // num_head, max_seq_len=max_seq_len, device=device, dtype=dtype
        )

        # Cache causal mask
        causal_mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor, train: bool) -> torch.Tensor:
        seq_len = x.shape[1]

        if not train:
            mask = None
        else:
            mask = self.causal_mask[:seq_len, :seq_len]

        qkv = self.project(x)
        q, k, v = einx.rearrange("b s (n h d) -> n b h s d", qkv, n=3, h=self.num_head)

        # Apply RoPE to q and k
        q = self.rope(einx.rearrange("b h s d -> b s h d", q))
        k = self.rope(einx.rearrange("b h s d -> b s h d", k))
        q = einx.rearrange("b s h d -> b h s d", q)
        k = einx.rearrange("b s h d -> b h s d", k)

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
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.rms_norm1 = RmsNorm(d_model, device=device, dtype=dtype)
        self.rms_norm2 = RmsNorm(d_model, device=device, dtype=dtype)
        self.mult_head_atten = MultiHeadAttention(
            d_model, num_heads, max_seq_len=max_seq_len, device=device, dtype=dtype
        )
        self.ffe = FFN(d_model, d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, train: bool) -> torch.Tensor:
        x_norm = self.rms_norm1(x)
        x_atten = self.mult_head_atten(x_norm, train)
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
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.embedding = Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype
        )
        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=max_seq_len,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RmsNorm(d_model=d_model, device=device, dtype=dtype)
        self.out_linear = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.max_seq_len = max_seq_len

    def forward(self, token_ids: torch.Tensor, train: bool) -> torch.Tensor:
        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x, train)
        x_norm = self.norm(x)
        logits = self.out_linear(x_norm)
        return logits


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax = Softmax()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Reshape for cross_entropy, handling any shape of logits
        logits = einx.rearrange("b... c -> (b...) c", logits)
        targets = einx.rearrange("b... -> (b...)", targets)

        probs = self.softmax(logits)

        correct_probs = einx.get_at("c[i], i -> ", probs, targets)
        # correct_probs = probs[targets]
        nll = -torch.log(correct_probs)
        mean_loss = einx.mean("... ->", nll)

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

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
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


ParamsT: TypeAlias = Union[
    Iterable[torch.Tensor], Iterable[Dict[str, Any]], Iterable[Tuple[str, torch.Tensor]]
]


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        weight_decay=1e-3,
        delta=1e-8,
    ):
        if lr < 0:
            raise ValueError(f"invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "weight_decay": weight_decay,
            "delta": delta,
        }
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            delta = group["delta"]
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
                p.data.addcdiv_(m_hat, torch.sqrt(sm_hat) + delta, value=-lr)

                # Weight decay
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

                state["t"] = t
        return loss


def cos_lr_scheduler(t_w: int, t_c: int, t: int, lr_max: float, lr_min: float) -> float:
    if t <= t_w:
        return lr_max * t / t_w
    elif t_w < t < t_c:
        return lr_min + 0.5 * (lr_max - lr_min) * (
            1 + math.cos(math.pi * (t - t_w) / (t_c - t_w))
        )
    else:
        return lr_min


def gradient_clip(params: Iterable[torch.Tensor], max_norm: float, delta=1e-6):
    with torch.no_grad():
        for param in params:
            if param.grad is None:
                continue
            grad_norm = torch.linalg.norm(param.grad.data)
            if grad_norm > max_norm:
                param.grad.data.mul_(max_norm / (grad_norm + delta))


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
