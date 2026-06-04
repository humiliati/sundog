"""Classical constraint-propagation + DFS Sudoku solver for the LDT build-gate
Amendment 01 ([I3'] process supervision, docs/lattice/PHASE1_AMENDMENT_01_I3_PROCESS_SUPERVISION.md).

Generates SOUND candidate-set narrowing trajectories for the LDT to imitate. The solver
is ANSWER-KEY-FREE: propagation and search read only the puzzle (the leakage guard, spec
§3.3). A *state* is a list of 81 ints; cand[c] is a 9-bit mask, bit d set <=> digit (d+1)
is a live candidate in cell c. Solved cell <=> popcount==1. Contradiction <=> some
cand[c]==0, or a unit where a digit has no legal home.

Each search node yields a training pair (L_t, L_fix, status): L_t = the lattice the model
would see (after the parent's tentative assignment, before deduction), L_fix = the sound
CP fixpoint from L_t (the imitation target), status in {ok, solved, contradiction}.

Run as __main__ for the soundness self-test (no dataset dependency).
"""
from __future__ import annotations

N = 9
N2 = 81
FULL = (1 << N) - 1  # 0b111111111 — all nine digits candidate


def _popcount(x: int) -> int:
    return bin(x).count("1")


# --- deterministic peers + units -------------------------------------------------
def _build_units():
    rows = [[r * N + c for c in range(N)] for r in range(N)]
    cols = [[r * N + c for r in range(N)] for c in range(N)]
    boxes = []
    for br in range(0, N, 3):
        for bc in range(0, N, 3):
            boxes.append([(br + dr) * N + (bc + dc) for dr in range(3) for dc in range(3)])
    return rows + cols + boxes


UNITS = _build_units()                      # 27 units (9 rows, 9 cols, 9 boxes)
_peers = [set() for _ in range(N2)]
for _u in UNITS:
    for _c in _u:
        _peers[_c].update(_u)
for _c in range(N2):
    _peers[_c].discard(_c)
PEERS = [frozenset(p) for p in _peers]      # 20 peers per cell


def puzzle_to_cand(grid):
    """grid: 81 ints (0=blank, 1..9=clue) -> initial candidate masks (no propagation)."""
    cand = [FULL] * N2
    for c in range(N2):
        v = grid[c]
        if v:
            cand[c] = 1 << (v - 1)
    return cand


def propagate(cand):
    """Sound CP to fixpoint on a COPY of cand. Returns (cand, status).
    Rules (all sound): peer-elimination from solved cells (covers naked singles) and
    hidden singles. Never removes a candidate that is consistent with the current state."""
    cand = list(cand)
    changed = True
    while changed:
        changed = False
        for c in range(N2):                 # naked singles -> peer elimination
            cc = cand[c]
            if cc == 0:
                return cand, "contradiction"
            if _popcount(cc) == 1:
                for p in PEERS[c]:
                    if cand[p] & cc:
                        cand[p] &= ~cc
                        changed = True
                        if cand[p] == 0:
                            return cand, "contradiction"
        for unit in UNITS:                  # hidden singles
            for d in range(N):
                bit = 1 << d
                homes = [c for c in unit if cand[c] & bit]
                if len(homes) == 0:
                    return cand, "contradiction"
                if len(homes) == 1:
                    c = homes[0]
                    if cand[c] != bit:
                        cand[c] = bit
                        changed = True
    status = "solved" if all(_popcount(cand[c]) == 1 for c in range(N2)) else "ok"
    return cand, status


def solve_and_log(cand_in, log, node_cap, counter):
    """DFS from cand_in (pre-CP). Appends (L_t, L_fix, status) for every node visited.
    Deterministic: branch on the most-constrained unsolved cell (popcount then index),
    digits ascending. Returns True if a solution exists in this subtree."""
    counter[0] += 1
    if counter[0] > node_cap:
        return False
    cand, status = propagate(cand_in)
    log.append((list(cand_in), cand, status))
    if status == "contradiction":
        return False
    if status == "solved":
        return True
    c = min((cc for cc in range(N2) if _popcount(cand[cc]) > 1),
            key=lambda cc: (_popcount(cand[cc]), cc))
    for d in range(N):
        bit = 1 << d
        if cand[c] & bit:
            child = list(cand)
            child[c] = bit                  # tentative assignment
            if solve_and_log(child, log, node_cap, counter):
                return True
    return False


def solve(grid, node_cap=4096):
    """Returns (solved: bool, solution_grid: list[81]|None, log: list[(L_t,L_fix,status)])."""
    log = []
    counter = [0]
    ok = solve_and_log(puzzle_to_cand(grid), log, node_cap, counter)
    sol = None
    if ok:
        for _lt, lf, st in reversed(log):
            if st == "solved":
                sol = [lf[c].bit_length() for c in range(N2)]   # bit (1<<d) -> digit d+1
                break
    return ok, sol, log


def is_valid_solution(sol) -> bool:
    if sol is None or any(sol[c] < 1 or sol[c] > N for c in range(N2)):
        return False
    return all(sorted(sol[c] for c in unit) == list(range(1, N + 1)) for unit in UNITS)


# --- soundness self-test ---------------------------------------------------------
if __name__ == "__main__":
    # AI Escargot (Arto Inkala) — a heavy-search puzzle; exercises DFS + backtracking.
    PUZZLE = "100007090030020008009600500005300900010080002600004000300000010040000007007000300"
    grid = [int(ch) for ch in PUZZLE]
    ok, sol, log = solve(grid)
    print(f"solved={ok}  valid_solution={is_valid_solution(sol)}  nodes_logged={len(log)}")

    # soundness audit on solution-CONSISTENT states only (states whose solved cells all
    # match the unique solution): the true digit must NEVER be eliminated there.
    assert ok and is_valid_solution(sol), "solver failed to produce a valid solution"
    sol_bit = [1 << (sol[c] - 1) for c in range(N2)]

    def consistent(state):
        return all(_popcount(state[c]) != 1 or state[c] == sol_bit[c] for c in range(N2))

    consistent_states = violations = 0
    for lt, lf, st in log:
        if consistent(lt):
            consistent_states += 1
            for c in range(N2):
                if (lt[c] & sol_bit[c]) and not (lf[c] & sol_bit[c]):
                    violations += 1     # true digit eliminated on a consistent state = UNSOUND
    print(f"solution_consistent_states={consistent_states}  soundness_violations={violations}")
    assert violations == 0, "CP eliminated a true digit on a solution-consistent state"
    # determinism: identical log length on a second solve
    _, _, log2 = solve(grid)
    print(f"deterministic_log_len={len(log) == len(log2)}")
    print("SOUNDNESS_OK")
