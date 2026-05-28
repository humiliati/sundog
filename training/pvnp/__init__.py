"""SUNDOG_V_P_V_NP Phase 1 training modules.

ML side of the pvnp-phase1 toy verifier:

- train_mlp_policy.py: small MLP policy trained by behavior cloning on
  the hand-coded safe-seeker traces.
- train_inversion_attacker.py: A_inv_small — predicts basin occupancy
  from sigma certificates.

Both modules export weights as portable JSON so the JS harness can load
them without a Python runtime dependency at evaluation time.
"""
