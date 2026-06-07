#!/usr/bin/env python
"""Path-B de-risk: GPU batched-rollout micro-benchmark for the three-body Phase-4 particle-MPC.

Faithfully ports ONLY the hot loop (computeAcceleration + RK4 integrateStep from
public/js/threebody-core.mjs) to batched torch, to confirm the 1080 speedup before committing the
full port. NOT the science harness; timing + a single-step sanity only.

Dynamics (verbatim from threebody-core.mjs lines 149-214):
  state = [x0,y0, x1,y1, x2,y2, vx0,vy0, vx1,vy1, vx2,vy2]  (3 bodies; 2=test particle, m3=0.01)
  a_i = sum_j G*m_j*(p_j - p_i)/r^3, with: primaries 0,1 IGNORE the test particle (index<2 & j==2
        skipped); test particle 2 feels 0 and 1 + thrust; softening: r<0.01 contributes 0.
  integrateStep = classic RK4 (4 accel evals/step).

    python scripts/proof_threebody_gpu_rollout_bench.py
"""
import time
import torch

G = 1.0
MASSES = torch.tensor([1.0, 1.0, 0.01])      # [m0, m1, m2]; m2 = test particle
SOFT = 0.01                                   # r < SOFT -> contribution skipped (== Node)


def accel(pos, thrust, masses):
    """pos (B,3,2), thrust (B,2) on the test particle -> acc (B,3,2). Faithful asymmetric rule."""
    m0, m1 = masses[0], masses[1]
    p0, p1, p2 = pos[:, 0], pos[:, 1], pos[:, 2]              # (B,2) each

    def term(pi, pj, mj):                                    # accel on i from j = G*mj*(pj-pi)/r^3
        d = pj - pi
        r = torch.linalg.vector_norm(d, dim=-1, keepdim=True)
        contrib = G * mj * d / (r ** 3)
        return torch.where(r < SOFT, torch.zeros_like(contrib), contrib)

    a = torch.zeros_like(pos)
    a[:, 0] = term(p0, p1, m1)                               # body 0 (primary) feels body 1 only
    a[:, 1] = term(p1, p0, m0)                               # body 1 (primary) feels body 0 only
    a[:, 2] = term(p2, p0, m0) + term(p2, p1, m1)            # test feels 0 and 1
    a[:, 2] = a[:, 2] + thrust
    return a


def rk4_step(state, dt, thrust, masses):
    """state (B,12) -> (B,12). RK4 matching integrateStep."""
    pos = state[:, :6].reshape(-1, 3, 2)
    vel = state[:, 6:].reshape(-1, 3, 2)

    def acc(p):
        return accel(p, thrust, masses)

    k1v, k1x = acc(pos), vel
    k2v, k2x = acc(pos + 0.5 * dt * k1x), vel + 0.5 * dt * k1v
    k3v, k3x = acc(pos + 0.5 * dt * k2x), vel + 0.5 * dt * k2v
    k4v, k4x = acc(pos + dt * k3x), vel + dt * k3v
    new_pos = pos + (dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x)
    new_vel = vel + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return torch.cat([new_pos.reshape(-1, 6), new_vel.reshape(-1, 6)], dim=1)


def rollout(B, horizon, dt, device):
    masses = MASSES.to(device)
    # init: primaries on x-axis, test particle near body 1, small per-trajectory jitter (belief cloud)
    torch.manual_seed(0)
    base = torch.tensor([-0.5, 0.0, 0.5, 0.0, 0.55, 0.1, 0.0, 0.7, 0.0, -0.7, 0.2, 0.3])
    state = base.to(device).repeat(B, 1) + 0.01 * torch.randn(B, 12, device=device)
    thrust = torch.zeros(B, 2, device=device)
    for _ in range(horizon):
        state = rk4_step(state, dt, thrust, masses)
    return state


def bench(device, B, horizon, dt, reps):
    if device == "cuda":
        torch.cuda.synchronize()
    # warmup
    rollout(min(B, 256), 16, dt, device)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        s = rollout(B, horizon, dt, device)
        if device == "cuda":
            torch.cuda.synchronize()
    el = (time.time() - t0) / reps
    rollouts_per_s = B / el
    step_evals_per_s = B * horizon / el
    return el, rollouts_per_s, step_evals_per_s, s


def main():
    dt = 0.01
    horizon = 800                               # = PINNED planningHorizonSteps (iad-shard)
    has_cuda = torch.cuda.is_available()
    print(f"torch {torch.__version__} cuda={has_cuda} "
          f"{torch.cuda.get_device_name(0) if has_cuda else 'cpu-only'}", flush=True)

    # single-step sanity: one trajectory, compare a hand value is finite + deterministic
    s1 = rk4_step(torch.tensor([[-0.5, 0., 0.5, 0., 0.55, 0.1, 0., 0.7, 0., -0.7, 0.2, 0.3]]),
                  dt, torch.zeros(1, 2), MASSES)
    print(f"[sanity] one RK4 step finite={bool(torch.isfinite(s1).all())} "
          f"x2->{s1[0,4].item():.6f},{s1[0,5].item():.6f}", flush=True)

    print(f"\n{'device':6} {'B':>6} {'horizon':>7} {'sec/rollout-batch':>18} {'rollouts/s':>12} {'RK4-step-evals/s':>16}")
    for B in (512, 5120):
        for device in (["cuda"] if has_cuda else []) + ["cpu"]:
            reps = 5 if device == "cuda" else 2
            el, rps, seps, _ = bench(device, B, horizon, dt, reps)
            print(f"{device:6} {B:6d} {horizon:7d} {el:18.4f} {rps:12.0f} {seps:16.3e}", flush=True)

    # Node reference: ~3.75 h/seed for 512 particles x 800 horizon x ~1600 control steps x ~candidates
    print("\nNode CPU reference (committed receipt): ~3.75 h/seed (serial particle x horizon loop).")
    print("If a 512-particle x 800-step rollout batch runs in ~ms on the 1080, the per-control-step")
    print("MPC (candidates x particles batched) is GPU-bound only by the 800 sequential steps ->")
    print("the full-port speedup is the rollouts/s ratio (GPU vs CPU) above.")


if __name__ == "__main__":
    main()
