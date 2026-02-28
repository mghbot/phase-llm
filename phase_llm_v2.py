"""
PhaseLLM v2 — Hybrid Phase Space + LLM AI
Single file. One command: python phase_llm_v2.py

Uses LOCAL Qwen3 0.6B at ~/phase-space-qwen3/
Falls back to lightweight random projection if model unavailable.
No downloads needed. Runs on CPU.

Requirements: pip install torch transformers sympy numpy scikit-learn scipy
"""

import torch
import torch.nn as nn
import numpy as np
import sympy
import time
import sys
import os
from dataclasses import dataclass
from typing import Any, Optional

from ops_library import build_phase_system, RealPhaseRouter, OpExecutor


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
# Falls back to hash-based projection if model unavailable
# ============================================================

class LLMInterface(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self._use_fallback = False

        if os.path.isdir(cfg.llm_model):
            from transformers import AutoTokenizer, AutoModel
            print(f"[LLM] Loading {cfg.llm_model}...")
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.backbone = AutoModel.from_pretrained(cfg.llm_model)
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

            hdim = self.backbone.config.hidden_size

            self.proj = nn.Sequential(
                nn.Linear(hdim, 256),
                nn.GELU(),
                nn.LayerNorm(256),
                nn.Linear(256, cfg.semantic_dim),
            )
            trainable = sum(p.numel() for p in self.proj.parameters())
            print(f"[LLM] Ready. Hidden dim: {hdim}. Projection: {trainable:,} trainable params")
        else:
            print(f"[LLM] Model not found at {cfg.llm_model}, using hash-based fallback")
            self._use_fallback = True
            # Deterministic projection from bag-of-chars to 64D
            self.fallback_proj = nn.Linear(256, cfg.semantic_dim)
            nn.init.orthogonal_(self.fallback_proj.weight)

    def encode(self, text):
        if isinstance(text, str):
            text = [text]

        if self._use_fallback:
            return self._fallback_encode(text)

        tok = self.tokenizer(text, max_length=self.cfg.max_tokens,
                             padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = self.backbone(**tok)
        pooled = out.last_hidden_state.mean(dim=1).float()
        return self.proj(pooled)

    def _fallback_encode(self, texts):
        """Hash-based deterministic encoding: text → 64D vector."""
        batch = []
        for text in texts:
            # Build a reproducible 256D feature vector from text content
            vec = np.zeros(256, dtype=np.float32)
            for i, ch in enumerate(text):
                idx = hash(ch) % 256
                vec[idx] += 1.0 / (1 + i * 0.01)
            # Normalize
            norm = np.linalg.norm(vec) + 1e-8
            vec = vec / norm
            batch.append(vec)
        x = torch.from_numpy(np.stack(batch))
        return self.fallback_proj(x)


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
#
# Wired to ops_library.py:
#   - build_phase_system() builds taxonomy, embeddings, router, executor
#   - RealPhaseRouter takes geo_embed(256) + sem_embed(64) as separate inputs
#   - OpExecutor runs SymPy operation chains from the 1,883 op taxonomy
# ============================================================

class PhaseLLM:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        print("=" * 50)
        print("  PhaseLLM v2 — Initializing")
        print("=" * 50)

        # Build phase system: taxonomy, Lorentz embeddings, router, executor
        self.ops, self.coords, self.router, self.executor = build_phase_system()
        self.cfg.num_ops = len(self.ops)

        self.llm = LLMInterface(self.cfg)
        self.fusion = FusionGate(self.cfg)
        self.validator = Validator()

        # Centroid of all op coordinates as initial geometric reference point
        self.geo_centroid = torch.from_numpy(
            self.coords.mean(axis=0).astype(np.float32)
        ).unsqueeze(0)  # (1, 256)

        print(f"[Fusion] {sum(p.numel() for p in self.fusion.parameters()):,} params")
        print(f"[Router] {self.router.param_count():,} trainable params")
        print("[PhaseLLM] Ready.\n")

    def solve(self, problem: str) -> SolveResult:
        t0 = time.perf_counter()

        # 1. Encode problem → 64D semantic embedding
        semantic = self.llm.encode(problem)  # (1, 64)

        # 2. Initial geometric embedding — centroid of operation space
        geometric = self.geo_centroid.clone()  # (1, 256)

        # 3. Fuse geometric + semantic → 256D fused coordinate
        fused = self.fusion(geometric, semantic)  # (1, 256)

        best_score, best_result, loops = 0.0, None, 0

        for i in range(self.cfg.max_refine + 1):
            # 4. Route: expand to top_k batch with perturbations for diversity
            top_k = self.cfg.top_k
            geo_batch = fused.detach().expand(top_k, -1) + torch.randn(top_k, self.cfg.geometric_dim) * 0.05
            sem_batch = semantic.detach().expand(top_k, -1)

            with torch.no_grad():
                traj_indices = self.router(geo_batch, sem_batch)  # list[list[int]]

            # 5. Execute each trajectory through OpExecutor
            results = []
            for op_ids in traj_indices:
                valid_ids = [idx for idx in op_ids if idx < len(self.ops)]
                op_names = [self.ops[idx]["name"] for idx in valid_ids]

                # Build Trajectory object
                if valid_ids:
                    traj_coords = self.coords[valid_ids]
                    energy = float(np.linalg.norm(traj_coords.mean(axis=0)))
                else:
                    traj_coords = np.zeros((0, 256), dtype=np.float32)
                    energy = 0.0

                traj = Trajectory(
                    op_ids=valid_ids,
                    op_names=op_names,
                    coordinates=traj_coords,
                    energy=energy,
                    confidence=1.0 / (1.0 + energy),
                )

                # Execute the operation chain via OpExecutor
                result_expr, success = self.executor.execute_chain(valid_ids, self.ops)
                results.append(ExecResult(
                    trajectory=traj,
                    result=result_expr,
                    result_str=str(result_expr),
                    success=success,
                ))

            # 6. Validate — pick best result
            score, result = self.validator.best(results, problem)
            if score > best_score:
                best_score, best_result = score, result
            if score >= self.cfg.confidence_threshold:
                break
            loops += 1

            # Re-fuse with perturbation for next refinement round
            geometric = self.geo_centroid + torch.randn_like(self.geo_centroid) * 0.1
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

    print("Done.")
