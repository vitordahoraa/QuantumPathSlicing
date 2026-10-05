# CLAUDE.md

## Project rule: the quantum part must be purely quantum

The quantum part of this project **cannot have any classical component**, except for **pre-processing**.

### Allowed (classical pre-processing only)
- Preparing, cleaning, normalizing, encoding or reducing the input data **before** it enters the quantum circuit/algorithm.
- Computing parameters needed to build the circuit (e.g. angles for state preparation or amplitude encoding) from the input, before execution.

### Not allowed in the quantum part
- Classical computation that substitutes for, shortcuts or assists the quantum algorithm (e.g. computing the answer classically and encoding it into the circuit).
- Classical optimization loops, classical post-processing that does the real work of the algorithm, or hybrid quantum-classical variants (e.g. VQE/QAOA-style classical optimizers), unless the user explicitly approves it.
- Classical fallbacks, simulations of the result by classical algorithms, or "oracles" that are implemented by solving the problem classically.

### How to apply
- Keep pre-processing code clearly separated from the quantum code (separate modules/functions), so the boundary is easy to audit.
- If a task seems to require a classical step inside or after the quantum part, **stop and ask the user** instead of adding it.
- Reading out measurement results is acceptable; any further classical processing of them that is essential to producing the answer should be flagged to the user.
