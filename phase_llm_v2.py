"""
PhaseLLM v2 — Hybrid Phase Space + LLM AI
Single file. One command: python phase_llm_v2.py

Uses LOCAL Qwen3 0.6B at ~/phase-space-qwen3/
No downloads needed. Runs on CPU.

Requirements: pip install torch transformers sympy numpy
"""

import torch
import torch.nn as nn
import numpy as np
import sympy
import time
import sys
from dataclasses import dataclass
from typing import Any, Optional


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    # LLM — local Qwen3 0.6B (1024 hidden dim, 28 layers)
    llm_model: str = "/home/michael/phase-space-qwen3"
    semantic_dim: int = 64
    max_tokens: int = 128
    # Fusion gate
    geometric_dim: int = 256
    fusion_hidden: int = 128
    alpha_init: float = 0.8
    alpha_floor: float = 0.6
    # PhaseRouter
    num_ops: int = 1883
    gru_hidden: int = 256
    gru_layers: int = 2
    beam_width: int = 16
    max_depth: int = 32
    # Pipeline
    top_k: int = 8
    max_refine: int = 3
    confidence_threshold: float = 0.7
    device: str = "cpu"


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Trajectory:
    op_ids: list
    op_names: list
    coordinates: np.ndarray
    energy: float
    confidence: float


@dataclass
class ExecResult:
    trajectory: Trajectory
    result: Any
    result_str: str
    success: bool
    error: Optional[str] = None


@dataclass
class SolveResult:
    problem: str
    answer: str
    confidence: float
    alpha: float
    time_ms: float
    refinements: int
    mode: str


# ============================================================
# LLM INTERFACE — Encodes NL → 64D semantic vector
# Uses local Qwen3 0.6B (1024 hidden → 64D)
# ============================================================

class LLMInterface(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        from transformers import AutoTokenizer, AutoModel
        print(f"[LLM] Loading {cfg.llm_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.backbone = AutoModel.from_pretrained(cfg.llm_model)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        hdim = self.backbone.config.hidden_size  # 1024 for Qwen3 0.6B

        # Projection: 1024 → 256 → 64
        # NO activation on final layer — preserves phase space geometry
        self.proj = nn.Sequential(
            nn.Linear(hdim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, cfg.semantic_dim),
            # No activation, no LayerNorm here — bare linear preserves geometry
        )
        trainable = sum(p.numel() for p in self.proj.parameters())
        print(f"[LLM] Ready. Hidden dim: {hdim}. Projection: {trainable:,} trainable params")

    def encode(self, text):
        if isinstance(text, str):
            text = [text]
        tok = self.tokenizer(text, max_length=self.cfg.max_tokens,
                             padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = self.backbone(**tok)
        pooled = out.last_hidden_state.mean(dim=1).float()
        return self.proj(pooled)


# ============================================================
# FUSION GATE — Learnable α blend of geometric + semantic
# ============================================================

class FusionGate(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.sem_proj = nn.Sequential(
            nn.Linear(cfg.semantic_dim, 128), nn.GELU(),
            nn.Linear(128, cfg.geometric_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(cfg.geometric_dim * 2, cfg.fusion_hidden), nn.GELU(),
            nn.Linear(cfg.fusion_hidden, 1),
        )
        nn.init.constant_(self.gate[-1].bias, 1.386)  # sigmoid(1.386) ≈ 0.8
        self._alpha = None

    def forward(self, geometric, semantic):
        sem_p = self.sem_proj(semantic)
        gate_in = torch.cat([geometric, sem_p], dim=-1)
        self._alpha = torch.sigmoid(self.gate(gate_in))
        return self._alpha * geometric + (1 - self._alpha) * sem_p

    @property
    def alpha(self):
        return self._alpha.mean().item() if self._alpha is not None else 0.0


# ============================================================
# PHASE ROUTER — STUB. Replace with your real GRU.
#
# To plug in your real system:
#   1. Replace self.gru / self.op_head with your trained GRU
#   2. Load your 1883x256 op embeddings via load_ops()
#   3. Rewrite navigate() with your beam search
#   4. Call freeze() after loading
# ============================================================

class PhaseRouter(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.Linear(768, 512), nn.GELU(),
            nn.Linear(512, cfg.geometric_dim), nn.LayerNorm(cfg.geometric_dim),
        )
        self.gru = nn.GRU(cfg.geometric_dim, cfg.gru_hidden,
                          cfg.gru_layers, batch_first=True)
        self.op_head = nn.Linear(cfg.gru_hidden, cfg.num_ops)
        self.op_embeddings = nn.Parameter(
            torch.randn(cfg.num_ops, cfg.geometric_dim) * 0.1,
            requires_grad=False)
        self.op_names = [f"op_{i}" for i in range(cfg.num_ops)]
        self.freeze()

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def navigate(self, coord, top_k=8):
        """STUB: Returns mock trajectories. Replace with your beam search."""
        trajs = []
        for _ in range(top_k):
            n = np.random.randint(3, min(15, self.cfg.max_depth))
            ids = np.random.choice(self.cfg.num_ops, n, replace=False).tolist()
            trajs.append(Trajectory(
                op_ids=ids,
                op_names=[self.op_names[i] for i in ids],
                coordinates=np.random.randn(n, self.cfg.geometric_dim).astype(np.float32),
                energy=float(np.random.exponential(1.0)),
                confidence=float(np.random.uniform(0.3, 0.95)),
            ))
        trajs.sort(key=lambda t: t.energy)
        return trajs[:top_k]

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        self.freeze()
        print(f"[PhaseRouter] Loaded from {path}, FROZEN")

    def load_ops(self, embeddings, names):
        self.op_embeddings = nn.Parameter(
            torch.from_numpy(embeddings).float(), requires_grad=False)
        self.op_names = names
        print(f"[PhaseRouter] Loaded {len(names)} operations")


# ============================================================
# OP ENGINE — STUB. Replace with your 1883 operations.
# ============================================================

class OpEngine:
    def __init__(self):
        x = sympy.Symbol('x')
        self.ops = {
            "integrate_poly": lambda e, v=x: sympy.integrate(e, v),
            "differentiate": lambda e, v=x: sympy.diff(e, v),
            "factor": lambda e: sympy.factor(e),
            "expand": lambda e: sympy.expand(e),
            "simplify": lambda e: sympy.simplify(e),
            "solve_eq": lambda e, v=x: sympy.solve(e, v),
        }

    def execute(self, traj, init_expr=None):
        try:
            cur = init_expr or sympy.Symbol('x')
            for name in traj.op_names:
                if name in self.ops:
                    cur = self.ops[name](cur)
            return ExecResult(traj, cur, str(cur), True)
        except Exception as e:
            return ExecResult(traj, None, "", False, str(e))

    def execute_top_k(self, trajs, k=3):
        return [self.execute(t) for t in trajs[:k]]


# ============================================================
# VALIDATOR
# ============================================================

class Validator:
    def score(self, result: ExecResult, problem: str = "") -> float:
        if not result.success or not result.result_str:
            return 0.0
        s = 0.5
        if len(result.result_str) > 10000:
            s -= 0.3
        if result.result_str.strip() == problem.strip():
            s -= 0.2
        math_chars = set("0123456789+-*/^()=xyzn")
        ratio = sum(1 for c in result.result_str if c in math_chars) / max(len(result.result_str), 1)
        s += ratio * 0.3
        return max(0.0, min(1.0, s)) * result.trajectory.confidence

    def best(self, results, problem=""):
        scored = [(self.score(r, problem), r) for r in results]
        scored.sort(key=lambda x: -x[0])
        return scored[0] if scored else (0.0, None)


# ============================================================
# PIPELINE — pipeline.solve("problem")
# ============================================================

class PhaseLLM:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        print("=" * 50)
        print("  PhaseLLM v2 — Initializing")
        print("=" * 50)
        self.llm = LLMInterface(self.cfg)
        self.fusion = FusionGate(self.cfg)
        self.router = PhaseRouter(self.cfg)
        self.engine = OpEngine()
        self.validator = Validator()
        print(f"[Fusion] {sum(p.numel() for p in self.fusion.parameters()):,} params")
        print("[PhaseLLM] Ready.\n")

    def solve(self, problem: str) -> SolveResult:
        t0 = time.perf_counter()

        semantic = self.llm.encode(problem)
        geometric = torch.randn(1, self.cfg.geometric_dim)  # STUB: replace w/ your encoder

        fused = self.fusion(geometric, semantic)

        best_score, best_result, loops = 0.0, None, 0
        for i in range(self.cfg.max_refine + 1):
            trajs = self.router.navigate(fused, self.cfg.top_k)
            results = self.engine.execute_top_k(trajs, k=self.cfg.top_k)
            score, result = self.validator.best(results, problem)
            if score > best_score:
                best_score, best_result = score, result
            if score >= self.cfg.confidence_threshold:
                break
            loops += 1
            fused = self.fusion(geometric, semantic)

        ms = (time.perf_counter() - t0) * 1000
        ans = best_result.result_str if best_result and best_result.success else "No solution"

        return SolveResult(
            problem=problem, answer=ans, confidence=best_score,
            alpha=self.fusion.alpha, time_ms=ms,
            refinements=loops, mode="hybrid",
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    p = PhaseLLM()

    problems = [
        "Find all primes p such that p^2 + 2 is also prime.",
        "Evaluate the integral of x^2 + 3x - 1 from 0 to 5.",
        "If f(x) = x^3 - 6x^2 + 11x - 6, find all roots of f.",
    ]

    for prob in problems:
        r = p.solve(prob)
        print(f"Problem:    {r.problem}")
        print(f"Answer:     {r.answer}")
        print(f"Confidence: {r.confidence:.4f}")
        print(f"Alpha:      {r.alpha:.4f} (1.0=pure geometric, 0.0=pure semantic)")
        print(f"Time:       {r.time_ms:.0f}ms | Refinements: {r.refinements}")
        print()

    print("Done. PhaseRouter + ops are stubs — plug in your real ones for real answers.")
