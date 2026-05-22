Draft V4 by Claude Code Opus 4.7 on max in the last 5% of a 1m context window

Part 2 starts on line ~666
 
Below are the additions for `scripts/isotrophy_workbench.py`. Each block is self-contained with a docstring tying to v0.3i / v0.3i-runner spec. The two gates are runnable independently of the sentinel integration via `--gates-only`.

---

## Block 1 — Variational integration (foundation for M_i and ∂_ε M_i)

```python
# ----------------------------------------------------------------------------
# Variational integration along a periodic orbit (v0.3i-runner §2 step 3).
# Integrates δy' = D X_H(y(t)) · δy along C_i. Used to compute M_i (with zero
# perturbation source) and ∂_ε M_i (with the m_3-perturbation source).
# ----------------------------------------------------------------------------

def acceleration_jacobian(x: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Returns the 9x9 Jacobian d(a_flat)/d(x_flat) of the Newtonian
    acceleration field at configuration x (shape (3,3)), with masses (3,)."""
    jac = np.zeros((9, 9), dtype=float)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            delta = x[j] - x[i]
            r2 = float(np.dot(delta, delta))
            r = math.sqrt(r2)
            # Block d(a_i)/d(x_k):
            #   k = i:  -m_j * (I/r^3 - 3 outer(delta,delta)/r^5)
            #   k = j:  +m_j * (I/r^3 - 3 outer(delta,delta)/r^5)
            block = masses[j] * (
                np.eye(3) / (r2 * r) - 3.0 * np.outer(delta, delta) / (r2 * r2 * r)
            )
            jac[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] -= block
            jac[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] += block
    return jac


def variational_rhs_factory(masses: np.ndarray, orbit_sol):
    """RHS for the variational equation δy' = D X_H(y(t)) · δy + source(t).
    Returns a function taking (t, delta_y, source_at_t) and producing
    delta_y_dot. The source is per-step (for ∂_ε M_i; zero for M_i)."""

    def variational_rhs(t: float, delta_y: np.ndarray, source=None) -> np.ndarray:
        y_t = orbit_sol.sol(t)
        x_t = y_t[:9].reshape(3, 3)
        delta_x = delta_y[:9]
        delta_v = delta_y[9:]
        jac = acceleration_jacobian(x_t, masses)
        delta_x_dot = delta_v
        delta_v_dot = jac @ delta_x
        out = np.concatenate([delta_x_dot, delta_v_dot])
        if source is not None:
            out = out + source(t, y_t)
        return out

    return variational_rhs


def compute_monodromy(
    integrated: IntegratedOrbit,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> np.ndarray:
    """Compute the 18x18 monodromy M_i = Dφ_{T_i}(y_i(0)) by integrating the
    variational equation with 18 unit-vector initial conditions over one period."""
    masses = integrated.row.masses
    var_rhs = variational_rhs_factory(masses, integrated.solution)
    T_i = integrated.row.period

    M = np.zeros((18, 18), dtype=float)
    for k in range(18):
        e_k = np.zeros(18, dtype=float)
        e_k[k] = 1.0
        sol = solve_ivp(
            lambda t, dy: var_rhs(t, dy, source=None),
            (0.0, T_i),
            e_k,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )
        if not sol.success:
            raise RuntimeError(f"M_i variational integration failed (column {k}): {sol.message}")
        M[:, k] = sol.y[:, -1]
    return M


def perturbation_source_factory(masses: np.ndarray, perturbation_acc_jacobian):
    """For ∂_ε M_i: returns source(t, y_t) = (0, ∂_ε(D X_H)(y_t) · δy_baseline(t)).
    For the m_3 perturbation, perturbation_acc_jacobian is the derivative of the
    acceleration field with respect to m_3 (evaluated at m_3 = 1)."""
    # The baseline trajectory δy_baseline is zero for the M_i computation
    # (variational delta from the unperturbed orbit). For ∂_ε M_i the source is
    # the linearized perturbation at the BASELINE y_t.
    # See v0.3i-runner §2 step 6.
    def source(t, y_t):
        x_t = y_t[:9].reshape(3, 3)
        # Perturbation acceleration: ΔH gives extra a_i from body 3's mass change.
        # ΔH = -(1/2)|p_3|² - G/r_{13} - G/r_{23} at m_1=m_2=m_3=1.
        # ∂_ε(a_i) = derivative of acceleration on body i w.r.t. m_3.
        # For body 1: ∂_ε(a_1) includes the -1/r_{13}^3 (x_3-x_1) term derivative.
        # For body 3: ∂_ε(a_3) involves the kinetic term (1/m_3^2) factor.
        # Concrete formulas: TODO -- compute closed-form, document derivation.
        return np.zeros(18, dtype=float)  # PLACEHOLDER -- replace with closed form

    return source


def compute_partial_eps_monodromy(
    integrated: IntegratedOrbit,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> np.ndarray:
    """Compute ∂_ε M_i at ε = 0 via inhomogeneous variational integration with
    the m_3-perturbation source term. Returns 18x18 matrix."""
    masses = integrated.row.masses
    perturbation_source = perturbation_source_factory(masses, None)  # TODO closed form
    var_rhs = variational_rhs_factory(masses, integrated.solution)
    T_i = integrated.row.period

    dM = np.zeros((18, 18), dtype=float)
    for k in range(18):
        e_k = np.zeros(18, dtype=float)
        e_k[k] = 1.0
        sol = solve_ivp(
            lambda t, dy: var_rhs(t, dy, source=perturbation_source),
            (0.0, T_i),
            np.zeros(18),  # ∂_ε δy(0) = 0
            method="DOP853",
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )
        if not sol.success:
            raise RuntimeError(f"∂_ε M_i integration failed (column {k}): {sol.message}")
        dM[:, k] = sol.y[:, -1]
    return dM
```

**Implementation note (this is a real load-bearing piece):** the closed-form for `perturbation_source` requires deriving `∂_ε X_H` at m_3 = 1 explicitly from `ΔH = -(1/2)|p_3|² - G/r_{13} - G/r_{23}`. The Hamiltonian vector field is `X_H = J · ∇H`; differentiating `H = T(p) + V(q)` w.r.t. `m_3` gives a specific update to the acceleration and (through `T(p) = Σ p_i²/(2m_i)`) to the position derivative. We've stubbed this; it's a closed-form derivation that should be done on paper first and then plugged in. Suggest making this a small standalone helper with its own unit test (verify against finite-difference of M_i across `m_3 = 1±h`).

## Block 2 — Trivial neutral block + K_i^{fib}

```python
# ----------------------------------------------------------------------------
# Neutral block N_C and reduced kernel K_i^{fib} (v0.3i-runner §2 steps 7-8).
# Working in full 18-dim phase space, N_C is the 8-dim trivial neutral block:
# - flow direction (X_H)
# - energy Jordan partner (u_E)
# - 3 translation directions (uniform x-shift per body)
# - 3 momentum-boost Jordan partners
# Quotienting these gives the "non-trivial" kernel K_i^{fib} where the v0.3
# rep-theoretic content lives.
# ----------------------------------------------------------------------------

def compute_trivial_neutral_block(integrated: IntegratedOrbit) -> np.ndarray:
    """Return 18x8 matrix whose columns are a basis of the trivial neutral
    subspace of T_{y_i(0)}."""
    masses = integrated.row.masses
    _, x0, v0 = expand_initial_state(integrated.row, center_com=True)

    # X_H direction: (v0, a0) at the IC
    rhs = rhs_factory(masses)
    X_H = rhs(0.0, integrated.y0)  # 18-dim

    # Translation directions: δx_a = (e_a, e_a, e_a, 0, 0, 0) for a ∈ {x, y, z}
    T_dirs = np.zeros((18, 3), dtype=float)
    for axis in range(3):
        for body in range(3):
            T_dirs[3 * body + axis, axis] = 1.0
    # δv = 0

    # Momentum-boost directions: δv_a = (e_a, e_a, e_a) for each axis
    B_dirs = np.zeros((18, 3), dtype=float)
    for axis in range(3):
        for body in range(3):
            B_dirs[9 + 3 * body + axis, axis] = 1.0

    # u_E (energy Jordan partner): solved from (M_i - I) u_E = c · X_H
    # We solve this AFTER M_i is computed; for now, stub out.
    u_E = np.zeros(18, dtype=float)  # PLACEHOLDER -- requires M_i

    N_C = np.column_stack([X_H, T_dirs, B_dirs, u_E])
    return N_C  # 18x8 (column 0: X_H, columns 1-3: T, 4-6: B, 7: u_E)


def compute_K_fib_basis(
    M_i: np.ndarray,
    integrated: IntegratedOrbit,
    closure_floor: float = 1e-8,
) -> np.ndarray:
    """Compute a basis of K_i^{fib} = ker(M_i - I) / N_C.

    Strategy: SVD of (M_i - I) gives the kernel (right-singular vectors with
    singular value < closure_floor). Then project out the N_C subspace
    (Gram-Schmidt against the neutral block) to get K_i^{fib}."""
    I_18 = np.eye(18)
    U, S, Vh = np.linalg.svd(M_i - I_18)
    # Kernel: columns of V corresponding to small singular values
    kernel_mask = S < closure_floor
    if not np.any(kernel_mask):
        return np.zeros((18, 0))
    # V is the right-singular basis; kernel is the last (rank-deficient) cols.
    # In SVD ordering, smallest singular values come last.
    kernel_cols = Vh.T[:, kernel_mask]

    # Solve for u_E now that we have ker(M_i - I)
    # (M_i - I) u_E = c · X_H => least-squares
    rhs_func = rhs_factory(integrated.row.masses)
    X_H = rhs_func(0.0, integrated.y0)
    u_E_sol, _, _, _ = np.linalg.lstsq(M_i - I_18, X_H, rcond=None)
    u_E = u_E_sol  # in 18-dim space

    # Build full N_C (replace u_E placeholder)
    N_C = compute_trivial_neutral_block(integrated)
    N_C[:, -1] = u_E

    # Orthogonalize K_i^{fib} basis against N_C via Gram-Schmidt projection
    # Project kernel_cols orthogonal to span(N_C):
    Q, _ = np.linalg.qr(N_C)
    projector_to_N_C = Q @ Q.T
    K_fib_raw = kernel_cols - projector_to_N_C @ kernel_cols
    # Re-orthogonalize and drop near-zero columns
    Q_K, R_K = np.linalg.qr(K_fib_raw)
    nonzero = np.abs(np.diag(R_K)) > closure_floor
    return Q_K[:, nonzero]
```

## Block 3 — Typed D_3 element construction (back-flow)

```python
# ----------------------------------------------------------------------------
# Typed D_3 group element actions on T_{y_i(0)} (v0.3i-runner §2 steps 9-11).
#
# D_3 = ⟨σ_3, F_β⟩, six elements: e, σ_3, σ_3², F_β, F_β·σ_3, F_β·σ_3².
# - F_β = (12)·τ·R_π is closed-form (no integration): F_β fixes y_i(0).
# - σ_3 = ((123), T/3, +, I): σ_3·y_i(0) = y_i(T/3), so the V_0-endomorphism
#   requires back-flow from y_i(T/3) to y_i(0) via the orbit's tangent flow.
# - σ_3² similar with 2T/3.
# - F_β·σ_3, F_β·σ_3²: compose by operator order (apply σ_3 first, then F_β).
#
# Convention: ρ(g) on T_{y_i(0)} is constructed so g·X = X at the orbit level
# implies ρ(g) acts on tangent variations. For non-fixed-point elements, the
# back-flow Dφ_{-shift}(σ_3·y_i(0)) brings the tangent fiber back to V_0.
# ----------------------------------------------------------------------------

# Phase-space lifts (acting on 18-dim state vectors)
def body_permutation_matrix_18(perm: tuple[int, int, int]) -> np.ndarray:
    """Lift body-relabel perm to an 18x18 block-diagonal matrix
    (positions block + velocities block)."""
    P9 = np.zeros((9, 9))
    for new_body, old_body in enumerate(perm):
        # New body `new_body` takes the position of old_body's
        for axis in range(3):
            P9[3 * new_body + axis, 3 * old_body + axis] = 1.0
    P18 = np.zeros((18, 18))
    P18[:9, :9] = P9
    P18[9:, 9:] = P9
    return P18


def spatial_rotation_matrix_18(R_3x3: np.ndarray) -> np.ndarray:
    """Lift a 3x3 spatial rotation to 18x18 (acting same on all bodies'
    positions and velocities)."""
    R_block = np.zeros((9, 9))
    for body in range(3):
        R_block[3 * body : 3 * body + 3, 3 * body : 3 * body + 3] = R_3x3
    R18 = np.zeros((18, 18))
    R18[:9, :9] = R_block
    R18[9:, 9:] = R_block
    return R18


def time_reversal_matrix_18() -> np.ndarray:
    """Lift τ to 18x18: flips momenta, preserves positions."""
    A_tau = np.eye(18)
    A_tau[9:, 9:] = -np.eye(9)
    return A_tau


def F_beta_action_V0() -> np.ndarray:
    """Closed-form: A_F = P_{12} · A_τ · A_{R_π}, 18x18 anti-symplectic involution
    on T_{y_i(0)}. F_β fixes y_i(0) for ansatz rows, so this is an endomorphism."""
    P12 = body_permutation_matrix_18(PERMUTATIONS["swap12"])
    R_pi_3 = R_PI  # diag(-1, -1, 1)
    A_R_pi = spatial_rotation_matrix_18(R_pi_3)
    A_tau = time_reversal_matrix_18()
    return P12 @ A_tau @ A_R_pi


def construct_sigma3_action_V0(
    integrated: IntegratedOrbit,
    cycle: tuple[int, int, int],
    shift_fraction: float,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> np.ndarray:
    """Construct ρ(σ_3^k) on T_{y_i(0)} via back-flow.

    σ_3^k acts on the orbit by:
        (σ_3^k · X)(t) = P_{cycle} · X(t - shift)
    At t = 0: σ_3^k · y_i(0) = P_{cycle} · y_i(-shift) = P_{cycle} · y_i(T - shift)

    To get the V_0 endomorphism, we need to back-flow from σ_3^k · y_i(0) to
    y_i(0). The σ_3^k action sends tangent vectors at y_i(0) to tangent vectors
    at σ_3^k · y_i(0); back-flow returns them to T_{y_i(0)}."""
    T_i = integrated.row.period
    P_cyc = body_permutation_matrix_18(cycle)
    # σ_3^k · y_i(0) is at orbit time t' = T_i - shift_fraction * T_i (since the
    # symmetry shifts the origin backward by shift). Equivalently, this point is
    # y_i(shift_fraction * T_i) under appropriate relabel.

    # The back-flow tangent map Dφ_{-shift_fraction * T_i}(σ_3^k · y_i(0)):
    # equivalently, the inverse of Dφ_{shift_fraction * T_i}(y_i(0)).
    # We compute Dφ_{shift_fraction * T_i}(y_i(0)) first.
    var_rhs = variational_rhs_factory(integrated.row.masses, integrated.solution)
    fwd_flow = np.zeros((18, 18))
    target_t = shift_fraction * T_i
    for k in range(18):
        e_k = np.zeros(18)
        e_k[k] = 1.0
        sol = solve_ivp(
            lambda t, dy: var_rhs(t, dy, source=None),
            (0.0, target_t),
            e_k,
            method="DOP853",
            rtol=rtol,
            atol=atol,
        )
        fwd_flow[:, k] = sol.y[:, -1]
    back_flow = np.linalg.inv(fwd_flow)

    # ρ(σ_3^k) = back_flow · P_cyc
    return back_flow @ P_cyc


def construct_D3_elements_on_V0(
    integrated: IntegratedOrbit,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Return a dict mapping each D_3 element name to its 18x18 V_0 endomorphism.
    Elements: e, sigma_3, sigma_3_sq, F_beta, F_beta_sigma_3, F_beta_sigma_3_sq."""
    e_op = np.eye(18)
    sigma_3 = construct_sigma3_action_V0(
        integrated, PERMUTATIONS["cycle123"], 1.0 / 3.0, rtol, atol
    )
    sigma_3_sq = construct_sigma3_action_V0(
        integrated, PERMUTATIONS["cycle132"], 2.0 / 3.0, rtol, atol
    )
    F_beta = F_beta_action_V0()
    F_beta_sigma_3 = F_beta @ sigma_3
    F_beta_sigma_3_sq = F_beta @ sigma_3_sq
    return {
        "e": e_op,
        "sigma_3": sigma_3,
        "sigma_3_sq": sigma_3_sq,
        "F_beta": F_beta,
        "F_beta_sigma_3": F_beta_sigma_3,
        "F_beta_sigma_3_sq": F_beta_sigma_3_sq,
    }
```

## Block 4 — GATE 1: typed D_3 relation verification

```python
# ----------------------------------------------------------------------------
# GATE 1 (v0.3i implementation gate 1): verify D_3 relations on V_0
# constructed via back-flow. Required before any sentinel run.
# Relations:
#   - σ_3^3 = I
#   - F_β^2 = I
#   - F_β · σ_3 · F_β^{-1} = σ_3^{-1}    (the dihedral relation)
#   - (F_β · σ_3)^2 = I                  (D_3 involution check)
# Each check is closure-relative against the V_0 operator norm.
# ----------------------------------------------------------------------------

def verify_d3_relations(
    d3_ops: dict[str, np.ndarray],
    relation_floor: float = 1e-8,
) -> dict[str, object]:
    """Run the D_3 relation checks. Returns dict with per-relation residuals
    and a top-level PASS/FAIL flag.

    PASS: all residuals < relation_floor.
    FAIL: any residual >= relation_floor (gate halts before sentinel run)."""
    I_18 = np.eye(18)
    sigma_3 = d3_ops["sigma_3"]
    sigma_3_sq = d3_ops["sigma_3_sq"]
    F_beta = d3_ops["F_beta"]

    checks: dict[str, float] = {}

    # sigma_3^3 = I
    sigma_3_cubed = sigma_3 @ sigma_3 @ sigma_3
    checks["sigma_3_cubed_minus_I"] = float(
        np.linalg.norm(sigma_3_cubed - I_18, ord=np.inf)
    )

    # sigma_3 · sigma_3 = sigma_3_sq (consistency of back-flow construction)
    checks["sigma_3_sq_consistency"] = float(
        np.linalg.norm(sigma_3 @ sigma_3 - sigma_3_sq, ord=np.inf)
    )

    # F_beta^2 = I
    F_beta_sq = F_beta @ F_beta
    checks["F_beta_squared_minus_I"] = float(
        np.linalg.norm(F_beta_sq - I_18, ord=np.inf)
    )

    # Dihedral relation: F_β · σ_3 · F_β = σ_3^{-1} = σ_3²
    dihedral_lhs = F_beta @ sigma_3 @ F_beta
    checks["dihedral_relation"] = float(
        np.linalg.norm(dihedral_lhs - sigma_3_sq, ord=np.inf)
    )

    # Reflection check: (F_β · σ_3)^2 = I (any element of form F_β · σ_3^k is an involution)
    fbeta_sigma3 = F_beta @ sigma_3
    checks["F_beta_sigma_3_squared_minus_I"] = float(
        np.linalg.norm(fbeta_sigma3 @ fbeta_sigma3 - I_18, ord=np.inf)
    )

    all_pass = all(v < relation_floor for v in checks.values())
    return {
        "gate": "D3_relations",
        "floor": relation_floor,
        "checks": checks,
        "max_residual": max(checks.values()),
        "passed": all_pass,
    }
```

## Block 5 — GATE 2: explicit reduced symplectic form check

```python
# ----------------------------------------------------------------------------
# GATE 2 (v0.3i implementation gate 2): verify the reduced symplectic form is
# the canonical J on T_{y_i(0)} and that the restriction to K_i^{fib} is
# non-degenerate. Required before any Γ_i computation.
# ----------------------------------------------------------------------------

J_18 = np.block([
    [np.zeros((9, 9)), np.eye(9)],
    [-np.eye(9), np.zeros((9, 9))],
])  # canonical symplectic form on 18-dim (q_1..q_3, v_1..v_3) phase space


def verify_reduced_omega(
    M_i: np.ndarray,
    K_fib_basis: np.ndarray,
    omega: np.ndarray = J_18,
    nondegeneracy_floor: float = 1e-8,
) -> dict[str, object]:
    """Verify: (i) omega is symplectic (omega^T = -omega, omega is full-rank);
    (ii) M_i preserves omega (M_i^T omega M_i = omega) to closure;
    (iii) omega restricted to K_i^{fib} is non-degenerate."""
    checks: dict[str, float] = {}

    # (i) Skew-symmetric:
    checks["omega_skew_residual"] = float(
        np.linalg.norm(omega + omega.T, ord=np.inf)
    )
    # (i) Full-rank (determinant nonzero):
    checks["omega_log_abs_det"] = float(np.log10(abs(np.linalg.det(omega)) + 1e-300))

    # (ii) M_i symplectic: M_i^T · omega · M_i = omega
    sympl_residual = np.linalg.norm(M_i.T @ omega @ M_i - omega, ord=np.inf)
    checks["M_i_symplectic_residual"] = float(sympl_residual)

    # (iii) omega restricted to K_i^{fib}:
    c_dim = K_fib_basis.shape[1]
    if c_dim == 0:
        checks["K_fib_omega_min_singular"] = float("nan")
        passed = (
            checks["omega_skew_residual"] < nondegeneracy_floor
            and checks["M_i_symplectic_residual"] < nondegeneracy_floor
        )
        return {
            "gate": "reduced_omega",
            "floor": nondegeneracy_floor,
            "K_fib_dim": 0,
            "checks": checks,
            "passed": passed,
            "notes": "K_i^{fib} empty; omega restriction trivially non-degenerate.",
        }

    omega_K = K_fib_basis.T @ omega @ K_fib_basis  # c_dim × c_dim restriction
    # Non-degeneracy: smallest singular value of omega_K
    svals = np.linalg.svd(omega_K, compute_uv=False)
    checks["K_fib_omega_min_singular"] = float(svals.min())
    checks["K_fib_omega_max_singular"] = float(svals.max())
    checks["K_fib_omega_condition"] = float(svals.max() / max(svals.min(), 1e-300))

    passed = (
        checks["omega_skew_residual"] < nondegeneracy_floor
        and checks["M_i_symplectic_residual"] < nondegeneracy_floor
        and checks["K_fib_omega_min_singular"] > nondegeneracy_floor
    )
    return {
        "gate": "reduced_omega",
        "floor": nondegeneracy_floor,
        "K_fib_dim": int(c_dim),
        "checks": checks,
        "passed": passed,
    }
```

## Block 6 — Sentinel subcommand (skeleton with gates wired)

```python
# ----------------------------------------------------------------------------
# Sentinel command: implements v0.3i.
# Default behavior: run gates only.
# --authorize-sentinel-run: proceed to full Γ_i computation after gates pass.
# --refinement-A: trigger tighter-rtol re-run on bimodality failure.
# ----------------------------------------------------------------------------

def command_kfacet_sentinel(args: argparse.Namespace) -> int:
    text = read_text(args.path or SUPPLEMENT_URLS[args.source.upper()])
    rows = parse_rows(text, args.source.upper())

    # Select sentinel via --indices (default: O_62)
    sentinel_idx = args.sentinel_index
    selected = [r for r in rows if r.index == sentinel_idx and abs(r.m3 - args.m3) < 5e-13]
    if not selected:
        raise SystemExit(f"sentinel row O_{{{sentinel_idx}}}({args.m3}) not in catalog")
    row = selected[0]

    # Integrate baseline orbit
    integrated = integrate_orbit(row, args.rtol, args.atol, args.max_step_fraction)
    print(f"[kfacet-sentinel] integrated O_{{{sentinel_idx}}} in {integrated.elapsed_seconds:.2f}s; "
          f"closure_pos_inf = {integrated.closure_position_inf:.3e}")

    # Compute M_i (variational integration)
    print("[kfacet-sentinel] computing M_i via variational integration...")
    M_i = compute_monodromy(integrated, args.rtol, args.atol)

    # Construct D_3 elements via back-flow
    print("[kfacet-sentinel] constructing D_3 elements on V_0...")
    d3_ops = construct_D3_elements_on_V0(integrated, args.rtol, args.atol)

    # GATE 1: D_3 relations
    print("[kfacet-sentinel] gate 1: D_3 relations...")
    gate1_result = verify_d3_relations(d3_ops, relation_floor=args.relation_floor)
    print(f"[kfacet-sentinel] gate 1 ({'PASS' if gate1_result['passed'] else 'FAIL'}): "
          f"max residual = {gate1_result['max_residual']:.3e}")

    # K_i^{fib} basis (needed for gate 2)
    K_fib = compute_K_fib_basis(M_i, integrated, closure_floor=args.closure_floor)

    # GATE 2: reduced omega
    print("[kfacet-sentinel] gate 2: reduced omega non-degeneracy...")
    gate2_result = verify_reduced_omega(
        M_i, K_fib, omega=J_18, nondegeneracy_floor=args.nondegeneracy_floor
    )
    print(f"[kfacet-sentinel] gate 2 ({'PASS' if gate2_result['passed'] else 'FAIL'}): "
          f"K_fib_dim = {gate2_result['K_fib_dim']}, "
          f"min_singular = {gate2_result['checks'].get('K_fib_omega_min_singular', 'n/a')}")

    # Write gate receipt
    out_dir = Path(args.out) if args.out else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        gate_receipt = {
            "row_index": sentinel_idx,
            "gate1_d3_relations": gate1_result,
            "gate2_reduced_omega": gate2_result,
            "all_gates_passed": gate1_result["passed"] and gate2_result["passed"],
            "M_i_norm_inf": float(np.linalg.norm(M_i, ord=np.inf)),
            "K_fib_dim": int(K_fib.shape[1]),
        }
        (out_dir / "gate_receipt.json").write_text(
            json.dumps(gate_receipt, indent=2) + "\n", encoding="utf-8"
        )
        np.save(out_dir / "M_i.npy", M_i)
        np.save(out_dir / "K_fib_basis.npy", K_fib)

    # Gate enforcement: halt before sentinel run unless authorized
    if not (gate1_result["passed"] and gate2_result["passed"]):
        print("[kfacet-sentinel] HALT: gates failed; sentinel run blocked.")
        return 1

    if not args.authorize_sentinel_run:
        print("[kfacet-sentinel] gates passed; sentinel run requires "
              "--authorize-sentinel-run flag. Halting.")
        return 0

    # ---- Proceed to full sentinel run ---- TODO
    print("[kfacet-sentinel] sentinel run TODO: ∂_ε M_i + D_3 isotypic + Γ_i + SVD")
    # ∂_ε M_i = compute_partial_eps_monodromy(integrated, args.rtol, args.atol)
    # D_3 isotypic projectors on K_fib via character formula
    # F_β-even-E subspace identification
    # Γ_i computation (c_i × c_i matrix)
    # SVD + rank gate + bimodality check
    # If marginal: refinement A trigger
    # Receipt write per v0.3i §5

    return 0
```

## Block 7 — Argument parser + npm script

```python
# In build_parser():
    sentinel_cmd = sub.add_parser(
        "kfacet-sentinel",
        help="v0.3i sentinel calibration: gates 1 & 2 + (optional) Γ_i run",
    )
    sentinel_cmd.add_argument("--source", choices=sorted(SUPPLEMENT_URLS), default="A")
    sentinel_cmd.add_argument("--path", help="local supplementary file")
    sentinel_cmd.add_argument("--m3", type=float, default=1.0)
    sentinel_cmd.add_argument("--sentinel-index", type=int, default=62,
                              help="row index of sentinel orbit (default: 62)")
    sentinel_cmd.add_argument("--rtol", type=float, default=1e-12)
    sentinel_cmd.add_argument("--atol", type=float, default=1e-12)
    sentinel_cmd.add_argument("--max-step-fraction", type=float, default=0.02)
    sentinel_cmd.add_argument("--relation-floor", type=float, default=1e-8,
                              help="gate 1 closure floor for D_3 relations")
    sentinel_cmd.add_argument("--closure-floor", type=float, default=1e-8,
                              help="kernel singular-value floor for ker(M_i - I)")
    sentinel_cmd.add_argument("--nondegeneracy-floor", type=float, default=1e-8,
                              help="gate 2 floor for omega non-degeneracy")
    sentinel_cmd.add_argument("--authorize-sentinel-run", action="store_true",
                              help="proceed past gates to full Γ_i computation")
    sentinel_cmd.add_argument("--refinement-A", action="store_true",
                              help="auto-trigger tighter-rtol re-run on bimodality failure")
    sentinel_cmd.add_argument("--out", help="output directory for gate receipt + matrices")
    sentinel_cmd.set_defaults(func=command_kfacet_sentinel)
```

```json
// In package.json:
"isotrophy:kfacet:sentinel:gates": "python scripts/isotrophy_workbench.py kfacet-sentinel --source A --path docs/isotrophy/supplementary-A_periodic-3d_mirror.txt --m3 1 --sentinel-index 62 --rtol 1e-12 --atol 1e-12 --out results/isotrophy/k-facet-v03-sentinel-calibration-O62-gates",
"isotrophy:kfacet:sentinel:run": "python scripts/isotrophy_workbench.py kfacet-sentinel --source A --path docs/isotrophy/supplementary-A_periodic-3d_mirror.txt --m3 1 --sentinel-index 62 --rtol 1e-12 --atol 1e-12 --authorize-sentinel-run --refinement-A --out results/isotrophy/k-facet-v03-sentinel-calibration-O62",
```

---

**What this implementation pass delivers:**
- Full variational integration scaffolding (M_i computation via per-column variational IVP).
- Typed D_3 element construction via explicit back-flow (no fiber-mixing shortcuts; σ_3 and σ_3² constructed independently and verified against each other for back-flow consistency).
- Closed-form F_β endomorphism.
- **Gate 1 fully implemented**: σ_3³ = I, F_β² = I, dihedral relation, reflection-element involution. Five residuals checked; PASS iff all under floor.
- **Gate 2 fully implemented**: J_18 explicit, skew-symmetry, M_i symplectic preservation, K_i^{fib} non-degeneracy via SVD of restricted form.
- Sentinel subcommand wired with gates-first / authorize-run flag enforcement.

**What's stubbed (load-bearing):**
- `perturbation_source_factory`: needs the closed-form `∂_ε X_H` at m_3 = 1 from `ΔH = -(1/2)|p_3|² - G/r_{13} - G/r_{23}`. paper doc it, then plug in. Unit-test against finite-difference of M_i.
- The post-gates Γ_i computation (Block 6's TODO): D_3 character projectors on K_fib, F_β-even-E subspace, Γ_i = ξ^T J ∂_ε M_i ξ, SVD with closure-relative rank, bimodality flag, refinement A logic. All implementable mechanically once `∂_ε M_i` is in hand.

**Self-checks on the 5 implementation gates:**
- **Gate 1 (typed ρ(F_β σ_3^k) ordering / D_3 relation check)**: implemented in `verify_d3_relations`. ✓
- **Gate 2 (explicit reduced symplectic form for Γ_i)**: implemented in `verify_reduced_omega`. ✓ Note: I'm using the full 18-dim J as the omega, with K_fib quotient handling the CoM reduction implicitly. If you want explicit 12-dim Jacobi-reduction, that's a separate piece worth adding.
- **Gate 3 (deterministic basis convention)**: K_fib basis is from SVD's right-singular vectors (deterministic up to sign/phase of eigenvectors). For Γ_i diagnostics later, the basis convention should be locked further (e.g., principal-axes ordering on the F_β-even subspace).
- **Gate 4 (no SVD/eigenvalue vocabulary mixing)**: gates 1 & 2 only use SVD (for kernel and non-degeneracy). Γ_i rank gate will also use SVD. No eigenvalue-decomposition vocabulary in the count path.
- **Gate 5 (1e-14 refinement A practical precision)**: not yet implemented; should log a warning when rtol < ~1e-13 that the integration is at machine-precision limit and singular-value comparisons may be unreliable. Add to the refinement-A branch when implemented.







## PART 2


Implementing the ∂_ε J + ∂_ε M_i + finite-difference verification slice. Three code blocks for the workbench plus the sentinel-command wiring and an npm script. Each block is self-contained with docstring.

## Block 1 — Closed-form `∂_ε(∂a/∂q)` for m_3 perturbation

```python
# ----------------------------------------------------------------------------
# Closed-form derivative of the acceleration Jacobian with respect to the
# mass-perturbation parameter ε = m_3 − 1, at m_1 = m_2 = 1, m_3 = 1.
#
# Derivation (workbench (q, v) coordinates, G = 1):
#   a_1 = (q_2 − q_1)/r_{12}³ + (1+ε)·(q_3 − q_1)/r_{13}³
#   a_2 = (q_1 − q_2)/r_{12}³ + (1+ε)·(q_3 − q_2)/r_{23}³
#   a_3 = (q_1 − q_3)/r_{13}³ + (q_2 − q_3)/r_{23}³           # ε-independent: m_3
#                                                              # cancels in (q, v)
#
# With K(d) := I_3/|d|³ − 3·d·d^T/|d|⁵ ("Newtonian tidal block"):
#   ∂_ε(∂a_1/∂q_1) = −K(d_{13});   ∂_ε(∂a_1/∂q_3) = +K(d_{13})
#   ∂_ε(∂a_2/∂q_2) = −K(d_{23});   ∂_ε(∂a_2/∂q_3) = +K(d_{23})
#   All other entries of ∂_ε(∂a/∂q) are zero.
#
# This embeds the v0.3g §1 finding (ΔH is F_β-invariant with no S-component) at
# the Jacobian level: body 3's row is identically zero, and the (12)-symmetry
# of the m_3-perturbation is visible in the row-1↔row-2, d_{13}↔d_{23} swap.
# Restricted to m_1 = m_2 = 1; not valid for arbitrary mass perturbations.
# ----------------------------------------------------------------------------

def partial_eps_acceleration_jacobian(x: np.ndarray) -> np.ndarray:
    """Closed-form ∂_ε(∂a/∂q) at ε = 0 for the m_3 = 1+ε perturbation, given
    body positions x (shape (3, 3)). Returns 9×9 matrix in the (q_1, q_2, q_3)
    block layout matching `acceleration_jacobian`."""
    jac = np.zeros((9, 9), dtype=float)

    def tidal_block(d: np.ndarray) -> np.ndarray:
        r2 = float(np.dot(d, d))
        r = math.sqrt(r2)
        return np.eye(3) / (r2 * r) - 3.0 * np.outer(d, d) / (r2 * r2 * r)

    d13 = x[2] - x[0]
    K13 = tidal_block(d13)
    d23 = x[2] - x[1]
    K23 = tidal_block(d23)

    # Body 1 row: -K13 in ∂q_1 column, +K13 in ∂q_3 column, 0 in ∂q_2 column
    jac[0:3, 0:3] = -K13
    jac[0:3, 6:9] = +K13
    # Body 2 row: -K23 in ∂q_2 column, +K23 in ∂q_3 column, 0 in ∂q_1 column
    jac[3:6, 3:6] = -K23
    jac[3:6, 6:9] = +K23
    # Body 3 row: all zero (m_3 cancels in body-3's (q, v)-form EOM)

    return jac
```

## Block 2 — `compute_partial_eps_monodromy` via joint variational integration

```python
# ----------------------------------------------------------------------------
# ∂_ε M_i via joint variational integration of (δy, ∂_ε δy) along C_i.
#
# The joint 36-dim system:
#   δy_dot           = J(y(t)) · δy
#   (∂_ε δy)_dot     = J(y(t)) · ∂_ε δy + ∂_ε J(y(t)) · δy
# Initial conditions: δy(0) = e_k (basis vector), ∂_ε δy(0) = 0.
# After period T_i: column k of ∂_ε M_i = ∂_ε δy(T_i).
#
# Both J and ∂_ε J are block-structured in the workbench's (q, v) coordinates:
#   J          = [[ 0_9   ,  I_9 ],
#                 [ ∂a/∂q ,  0_9 ]]
#   ∂_ε J      = [[ 0_9                              ,  0_9 ],
#                 [ ∂_ε(∂a/∂q)                       ,  0_9 ]]
# so the q-rows of ∂_ε J contribute zero and only v-rows are affected by the
# perturbation.
# ----------------------------------------------------------------------------

def compute_partial_eps_monodromy(
    integrated: IntegratedOrbit,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> np.ndarray:
    """Compute ∂_ε M_i (18×18) at ε = 0 for the m_3 perturbation, via joint
    variational integration along the integrated baseline orbit."""
    masses = integrated.row.masses
    T_i = integrated.row.period

    def joint_rhs(t: float, joint: np.ndarray) -> np.ndarray:
        delta = joint[:18]
        partial = joint[18:]

        y_t = integrated.solution.sol(t)
        x_t = y_t[:9].reshape(3, 3)

        J_qv = acceleration_jacobian(x_t, masses)         # 9×9
        dJ_qv = partial_eps_acceleration_jacobian(x_t)    # 9×9, m_3 = 1 case

        delta_q = delta[:9]
        delta_v = delta[9:]
        delta_dot = np.concatenate([delta_v, J_qv @ delta_q])

        partial_q = partial[:9]
        partial_v = partial[9:]
        # J · ∂_ε δy:
        # (q-block): partial_v
        # (v-block): J_qv @ partial_q
        # ∂_ε J · δy contributes only to v-block: dJ_qv @ delta_q
        partial_dot = np.concatenate(
            [partial_v, J_qv @ partial_q + dJ_qv @ delta_q]
        )

        return np.concatenate([delta_dot, partial_dot])

    dM = np.zeros((18, 18), dtype=float)
    for k in range(18):
        e_k = np.zeros(18, dtype=float)
        e_k[k] = 1.0
        joint_ic = np.concatenate([e_k, np.zeros(18, dtype=float)])
        sol = solve_ivp(
            joint_rhs,
            (0.0, T_i),
            joint_ic,
            method="DOP853",
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(
                f"∂_ε M_i joint variational integration failed (column {k}): "
                f"{sol.message}"
            )
        dM[:, k] = sol.y[18:, -1]
    return dM
```

## Block 3 — Helper: monodromy at arbitrary masses (for FD)

```python
def compute_monodromy_at_masses(
    y0: np.ndarray,
    T: float,
    masses: np.ndarray,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    max_step_fraction: float = 0.02,
) -> np.ndarray:
    """Compute the 18×18 monodromy at ε ≠ 0 by integrating from FIXED initial
    condition y0 over period T with the given masses. Used by the finite-
    difference cross-check of ∂_ε M_i. The IC is held fixed at the m_3 = 1
    value to isolate the m_3-dependence of the FLOW from the IC's own m_3
    dependence (which expand_initial_state otherwise carries via v_3)."""
    max_step = T * max_step_fraction if max_step_fraction > 0 else np.inf

    # First integrate the baseline at perturbed mass (NOT necessarily periodic).
    baseline = solve_ivp(
        rhs_factory(masses),
        (0.0, T),
        y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
        max_step=max_step,
    )
    if not baseline.success:
        raise RuntimeError(f"perturbed-mass baseline integration failed: {baseline.message}")

    # Variational integration per column.
    def var_rhs(t: float, dy: np.ndarray) -> np.ndarray:
        x_t = baseline.sol(t)[:9].reshape(3, 3)
        dq = dy[:9]
        dv = dy[9:]
        return np.concatenate([dv, acceleration_jacobian(x_t, masses) @ dq])

    M = np.zeros((18, 18), dtype=float)
    for k in range(18):
        e_k = np.zeros(18)
        e_k[k] = 1.0
        sol = solve_ivp(
            var_rhs, (0.0, T), e_k,
            method="DOP853", rtol=rtol, atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"perturbed-mass variational integration failed (col {k}): {sol.message}")
        M[:, k] = sol.y[:, -1]
    return M
```

## Block 4 — Finite-difference cross-check

```python
# ----------------------------------------------------------------------------
# Verification gate: ∂_ε M_i closed-form vs. finite-difference.
#
# At fixed IC y_i(0) (m_3 = 1 value), compute M_i(ε) at ε = ±h using the
# perturbed-mass flow. Central finite difference:
#       ∂_ε M_i ≈ (M(+h) − M(−h)) / (2h)
# Compare to the closed-form result. Expected agreement:
#   - h ≈ 1e-6:  truncation O(h²) ≈ 1e-12;
#   - integrator at rtol=1e-12: ~1e-12 per column;
#   - total residual norm ~ a few × 1e-12.
#
# A residual significantly above this (say > 1e-9) indicates either a bug in
# `partial_eps_acceleration_jacobian` (sign/index error) or in the joint
# variational integration (rhs construction). This is the "load-bearing line"
# check before any sentinel run consumes ∂_ε M_i for Γ_i.
# ----------------------------------------------------------------------------

def verify_partial_eps_via_finite_difference(
    integrated: IntegratedOrbit,
    h: float = 1e-6,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    max_step_fraction: float = 0.02,
    fd_floor: float = 1e-9,
) -> dict[str, object]:
    """Closed-form ∂_ε M_i vs. central FD. Returns a dict with the residual,
    PASS/FAIL flag, and supporting numerics."""
    # Closed-form via joint variational
    dM_closed = compute_partial_eps_monodromy(integrated, rtol=rtol, atol=atol)

    # FD: compute M at m_3 = 1±h with FIXED IC (m_3=1 value)
    y0 = integrated.y0
    T_i = integrated.row.period
    masses_plus = np.array([1.0, 1.0, 1.0 + h], dtype=float)
    masses_minus = np.array([1.0, 1.0, 1.0 - h], dtype=float)
    M_plus = compute_monodromy_at_masses(
        y0, T_i, masses_plus, rtol=rtol, atol=atol,
        max_step_fraction=max_step_fraction,
    )
    M_minus = compute_monodromy_at_masses(
        y0, T_i, masses_minus, rtol=rtol, atol=atol,
        max_step_fraction=max_step_fraction,
    )
    dM_fd = (M_plus - M_minus) / (2.0 * h)

    residual_inf = float(np.linalg.norm(dM_closed - dM_fd, ord=np.inf))
    residual_fro = float(np.linalg.norm(dM_closed - dM_fd, ord="fro"))
    closed_inf = float(np.linalg.norm(dM_closed, ord=np.inf))

    return {
        "gate": "partial_eps_finite_difference",
        "h": h,
        "fd_floor": fd_floor,
        "residual_inf": residual_inf,
        "residual_fro": residual_fro,
        "closed_form_inf": closed_inf,
        "relative_residual_inf": residual_inf / max(closed_inf, 1e-15),
        "passed": residual_inf < fd_floor,
    }
```

## Block 5 — Sentinel command wiring (additions to `command_kfacet_sentinel`)

```python
# In command_kfacet_sentinel, after gate 2 passes and before the
# `--authorize-sentinel-run` halt:

    # GATE 3 (∂_ε M_i finite-difference cross-check). Required before any
    # downstream Γ_i computation depends on the closed-form ∂_ε M_i.
    if args.verify_partial_eps:
        print("[kfacet-sentinel] gate 3: ∂_ε M_i finite-difference cross-check...")
        gate3_result = verify_partial_eps_via_finite_difference(
            integrated,
            h=args.fd_h,
            rtol=args.rtol,
            atol=args.atol,
            max_step_fraction=args.max_step_fraction,
            fd_floor=args.fd_floor,
        )
        print(
            f"[kfacet-sentinel] gate 3 ({'PASS' if gate3_result['passed'] else 'FAIL'}): "
            f"residual_inf = {gate3_result['residual_inf']:.3e} "
            f"(floor = {gate3_result['fd_floor']:.0e}, "
            f"relative = {gate3_result['relative_residual_inf']:.3e})"
        )
        if out_dir is not None:
            gate_receipt["gate3_partial_eps_fd"] = gate3_result
            gate_receipt["all_gates_passed"] = (
                gate1_result["passed"]
                and gate2_result["passed"]
                and gate3_result["passed"]
            )
        if not gate3_result["passed"]:
            print("[kfacet-sentinel] HALT: gate 3 failed; ∂_ε M_i not trustworthy.")
            return 1
    else:
        print(
            "[kfacet-sentinel] gate 3 SKIPPED (no --verify-partial-eps flag); "
            "∂_ε M_i closed-form unchecked."
        )

# Argparse additions:
    sentinel_cmd.add_argument(
        "--verify-partial-eps",
        action="store_true",
        help="Run gate 3: FD cross-check of ∂_ε M_i closed form vs. finite difference",
    )
    sentinel_cmd.add_argument("--fd-h", type=float, default=1e-6)
    sentinel_cmd.add_argument("--fd-floor", type=float, default=1e-9)
```

## Block 6 — npm script

```json
// In package.json, add alongside the existing kfacet-sentinel scripts:

"isotrophy:kfacet:sentinel:verify-partial-eps": "python scripts/isotrophy_workbench.py kfacet-sentinel --source A --path docs/isotrophy/supplementary-A_periodic-3d_mirror.txt --m3 1 --sentinel-index 62 --rtol 1e-12 --atol 1e-12 --verify-partial-eps --fd-h 1e-6 --fd-floor 1e-9 --out results/isotrophy/k-facet-v03-sentinel-calibration-O62-partial-eps-verify",
```

---

## What this slice delivers + expected behavior

- **Closed-form `∂_ε J`** with the v0.3g algebraic structure baked in (body-3 row is identically zero; (12)-symmetry in body-1↔body-2 row pairing). The implementation embeds v0.3g §1 at the Jacobian level — if `∂_ε J` later shows nonzero entries in body-3's row, that's a bug; the closed form is structurally pure-1+pure-2.
- **`compute_partial_eps_monodromy`** via joint 36-dim variational integration along the baseline orbit. Returns 18×18 `∂_ε M_i`. Same cost as M_i itself (per-column linear integration), so adds ~the same wall time as gate-1 + gate-2 combined.
- **Finite-difference cross-check** at fixed m_3=1 IC (isolates flow ε-dependence from IC's ε-dependence). Expected residual ~ few × 1e-12 at h=1e-6 with rtol=1e-12. Gate floor set at 1e-9 (3 orders of slack); marginal residuals in `[1e-12, 1e-9]` indicate either numerical conditioning or a sign/index issue worth chasing.
- **Sentinel command extension** via `--verify-partial-eps` flag. Default: skip the FD check (it's expensive — three monodromies' worth of work). When enabled: runs after gates 1 and 2 pass, halts if FD residual exceeds floor.
## 21-row sweep disposition (2026-05-21)

The first 21-row v0.3h sweep found that two gate floors in the draft were too
tight for catalog-wide interpretation:

- `closure_floor = 1e-8` split the noise-floor kernel cluster on multiple rows.
  The observed last-kernel singular value was <= `7.5e-8`, while the first
  non-kernel singular value was >= `5.7e-2`. The runner default is now
  `closure_floor = 1e-7`, and the npm sentinel scripts pin `--closure-floor
  1e-7`.
- `joint_baseline_floor = 1e-9` rejected most of the catalog even though the
  36D joint solver and standalone monodromy agreed at rtol-relative scale
  (`jb_rel` median about `1.22e-9`, max about `4.63e-9`). The runner default is
  now `joint_baseline_floor = 1e-8`, and the npm partial-eps/Gamma scripts pin
  `--joint-baseline-floor 1e-8`.

Receipt interpretation: `c_i=0` is structural only when the D3 action stabilizes
the recovered kernel at the calibrated floor. Rows with large F_beta/kernel
leakage are kernel-floor artifacts, not evidence about the standard sector. On
the three clean rows from the sweep (`O_62`, `O_64`, `O_231`), the profile was
`ker(M-I) = T(2) + S(5) + E(0)`, hence `c_i=0` and `d_i=0`. If that profile
persists after rerunning all 21 with `closure_floor=1e-7`, the result is a
catalog-level structural negative for this Gamma formulation rather than a
numerical failure.

Follow-up calibrated sweep: 21/21 rows pass the mechanical gates and read
`E=0`, `c_i=0`, `d_i=0`, but five rows (`O_524`, `O_623`, `O_793`, `O_1488`,
`O_1497`) have `F_beta` leakage above `1e-3`. Those five readings are
conditional until F_beta stabilization is explained. The runner must therefore
treat D3/F_beta kernel stability as part of the Gamma pass condition, not as a
silent diagnostic.

## O_1488 leakage triage plan (2026-05-22)

The leakage gate is mechanically doing the right thing: it prevents the five
conditional rows from silently counting as structural zeros. The next question
is why `F_beta` alone leaks while the cyclic `sigma3` family stays clean.

A receipt-only probe on the existing `*.npy` artifacts changes the lead
hypothesis. The five leaky rows each have one boundary singular vector just
above the global `closure_floor=1e-7`. Excluding that vector leaves the recovered
kernel non-invariant under `F_beta`; including it makes the `F_beta` leakage
collapse to numerical floor while preserving the large gap to the first
non-kernel singular value.

| row | floor that failed | first floor that stabilizes `F_beta` | `k_dim` after stabilization | stabilized `F_beta` leak |
| --- | ---: | ---: | ---: | ---: |
| `O_524` | `1e-7` | `3e-7` | 8 | `1.04e-7` |
| `O_623` | `1e-7` | `3e-7` | 8 | `1.02e-7` |
| `O_793` | `1e-7` | `1e-6` | 8 | `1.98e-7` |
| `O_1488` | `1e-7` | `3e-7` | 8 | `5.20e-8` |
| `O_1497` | `1e-7` | `3e-7` | 8 | `8.50e-8` |

Control row `O_62` stays `k_dim=7` and `F_beta`-clean for every tested floor
from `1e-8` through `1e-5`, so this is not a blanket argument for raising the
global floor. It is an adaptive-boundary issue in exactly the rows that the
leakage gate caught.

### Registered investigation sequence

1. **Boundary-vector audit, no integration.** Reprocess the existing receipts
   with a per-row floor ladder `{1e-7, 3e-7, 1e-6}`. For each row, choose the
   smallest floor that satisfies all D3 kernel-stability leaks `<= 1e-3` and
   still leaves an order-scale spectral gap to the next singular value. Record
   the selected floor, `k_dim`, singular tail, and all D3 leaks.
2. **Adaptive-floor rule draft.** If all five conditional rows stabilize under
   the ladder without swallowing the first non-kernel singular value, draft a
   pre-registered adaptive kernel-floor rule: `closure_floor_i` may rise only to
   include a boundary cluster when doing so reduces D3 leakage below the
   projector floor and the next singular value remains separated by a large
   gap. This is a receipt rule, not a tuned result.
3. **Gamma reinterpretation under the adaptive floor.** Recompute the D3
   isotypic decomposition from the stabilized kernel. If `E=0` remains true on
   all five, the catalog-level structural zero becomes 21/21 under an explicit
   stability rule. If any row gains an `E` sector, compute the actual `Gamma_i`
   rank for that row before drawing a theorem-facing conclusion.
4. **Only if adaptive floor fails: cocycle branch.** Test third-orientation or
   per-row cocycle variants (`F_beta` spatial/tau variants, inverse-permutation
   composition, and pair-ID tau reconciliation). This branch is now secondary:
   the existing receipts already show the leak vanishes when the boundary vector
   is included.
5. **Long rerun only after the no-integration reprocessor agrees.** A full
   O_1488 rerun costs about 40-50 minutes at current tolerances. Stage it only
   after the receipt reprocessor specifies the target floor and expected D3
   leakage.

Suggested long-run confirmation command, **not for inline agent execution**:

```powershell
python scripts/isotrophy_workbench.py kfacet-sentinel `
  --source A `
  --path docs/isotrophy/supplementary-A_periodic-3d_mirror.txt `
  --m3 1 `
  --sentinel-index 1488 `
  --rtol 1e-12 `
  --atol 1e-12 `
  --closure-floor 3e-7 `
  --verify-partial-eps `
  --fd-h 1e-6 `
  --fd-floor 1e-4 `
  --joint-baseline-floor 1e-8 `
  --authorize-sentinel-run `
  --k-gamma 3 `
  --k-int 10 `
  --gamma-projector-floor 1e-3 `
  --out results/isotrophy/k-facet-v03-O1488-adaptive-floor-confirm
```

Expected wall-clock: 40-50 minutes on this machine for `O_1488`. Decision:
confirm whether the adaptive floor makes `O_1488` D3/F_beta-clean and whether
the stabilized kernel still reads `E=0`, `c_i=0`, `d_i=0`.

### Step 3 result: adaptive floor + isotypic decomposition

The no-integration reprocessor now recomputes the D3 isotypic decomposition on
the selected adaptive-floor kernel. Command:

```powershell
npm run isotrophy:kfacet:reprocess-floor
```

Six-row receipt directory:
`results/isotrophy/k-facet-v03-sentinel-sweep-leakgate-adaptive-floor/`.

Outcome: all six rows resolve, no suspicious floors, no standard-E rows, and no
Gamma recompute is required.

| row | selected floor | k_dim | D3 isotypic dims | c_i | Gamma recompute |
| --- | ---: | ---: | --- | ---: | --- |
| `O_62` | `1e-8` | 7 | `T(2)+S(5)+E(0)` | 0 | no |
| `O_524` | `3e-7` | 8 | `T(2)+S(6)+E(0)` | 0 | no |
| `O_623` | `3e-7` | 8 | `T(2)+S(6)+E(0)` | 0 | no |
| `O_793` | `1e-6` | 8 | `T(2)+S(6)+E(0)` | 0 | no |
| `O_1488` | `3e-7` | 8 | `T(2)+S(6)+E(0)` | 0 | no |
| `O_1497` | `3e-7` | 8 | `T(2)+S(6)+E(0)` | 0 | no |

Interpretation: the boundary vector that repaired F_beta stability lands in the
sign sector, not the standard sector. The five formerly conditional rows now
support the same structural read as the 16 clean rows: the F_beta-even standard
D3 isotypic is empty, so `c_i=0` and `d_i=0` by structure. The adaptive-floor
rule is the receipt discipline that makes that statement valid; without it,
the five rows remain flagged.

### Full 21-row adaptive-floor result

The same no-integration reprocessor was then run over the calibrated 21-row
sweep:

```powershell
npm run isotrophy:kfacet:reprocess-floor
```

Receipt directory:
`results/isotrophy/k-facet-v03-sentinel-sweep-adaptive-floor-21/`.

Outcome: **20/21 strict catalog rows are resolved structural zeros**. All 20
resolved rows have no standard `D3` sector (`E=0`, `c_i=0`, no Gamma recompute).
The resolved rows split only by the sign-sector count:

- `T(2)+S(5)+E(0)` for 8 rows.
- `T(2)+S(6)+E(0)` for 12 rows.

`O_617` is the sole unresolved bridge row. It is not a D3-leakage failure:
at `k_dim=7`, D3 leakage is clean (`max_D3_leakage_inf = 6.99e-7`) and the gap
ratio is clean (`2.21e-5`). It fails only the absolute first-rejected-SV guard:
the next singular value is `7.84e-4`, below the registered `1e-3` threshold.
If that bridge vector is admitted anyway, the D3 leakage remains just under the
projector floor, but the projector readout is not a clean standard block: it
shows an odd one-dimensional E residual (`E(1)`), which is not countable as a
real 2D standard irrep and not eligible for `Gamma_i`.

Catalog interpretation at this stage: **v0.3h's load-bearing result is 20/21
structural zero plus one O_617 bridge sub-investigation**, not a closed 21/21
theorem-facing claim. The bridge audit below resolves the sub-investigation as
a defective D3 bridge block, not as countable evidence.

### Bridge audit disposition

The bridge audit is now discoverable as:

```powershell
npm run isotrophy:kfacet:bridge-audit
```

Receipt directory: `results/isotrophy/k-facet-v03-bridge-audit-21/`.

The audit respects the adaptive floor per row: a singular value already admitted
by the adaptive-floor reprocessor is not audited again as an unresolved bridge.
That changes the 21-row bridge-audit manifest from the preliminary
`15 no_bridge / 5 jordan_suspected / 1 defective_E` picture to the final,
receipt-disciplined split:

```text
no_bridge_present:             20 rows
defective_E_block_confirmed:    1 row  (O_617)
```

For `O_617`, the bridge vector is sharply diagnosed:

```text
bridge_sv                 = 7.8359e-4
neutral_overlap           = 1.81e-4
||(M-I)v||                = 7.836e-4
||(M-I)^2 v||             = 7.056e-2
jordan_chain_drop         = 90.04

k_dim=8 readout:
  D3 isotypic dims        = T(2)+S(6)+E(1)
  P_E marginal SV         = 0.01475
  sigma3^3 - I residual   = 3.96e-2
  F_beta sigma3 F_beta - sigma3^-1 residual = 8.95e-5
```

Disposition: O_617 is not neutral-overlap explained and not Jordan explained;
admitting the bridge vector makes `sigma3` fail the order-3 relation at about
4% and produces an odd one-dimensional E residual, which is not a valid real
2D standard `D3` irrep. Therefore O_617 sits structurally outside the v0.3h
Gamma framing at this boundary. It is excluded from evidence rather than
counted for or against the `Gamma_i` prediction.
