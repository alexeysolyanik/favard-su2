# Favard's theorem on SU(2) and quantum computing: code

Scripts that reproduce the numerical values in the paper
*Favard's Theorem on SU(2) and Quantum Computing* by Alexey Solyanik.

Requirements: Python 3, NumPy, SciPy, mpmath (`pip install numpy scipy mpmath`).
Run each script from this folder, for example `python3 hk.py`.

| Script | What it computes | Where in the paper |
|---|---|---|
| `hk.py` | Sharp constants h_1, …, h_7 from the node equations (40-digit arithmetic); state constants h_k^st = H(k,S^2)/4 | Table 1 (sharp linear error), table of state constants |
| `designs.py` | Binary polyhedral groups (Pauli, tetrahedral, Clifford, icosahedral) and the dimensions of their fixed spaces; design strength t | Table of rotation groups as unitary designs |
| `econst.py` | Monte Carlo estimate of the exact worst-case factor e(G) for each group | Remark after the design theorem (e(G) about one quarter below h_{t+1}) |
| `testfun.py` | Errors of the group averages on the test response f_0 | Last column of the same table |
| `casimir.py` | Check of −Δf = 4N f for f(U) = Tr(O U ρ U*) on SU(N) | Normalization of the Laplacian on SU(N) |
| `compression_example.py` | Four-qubit Ising Trotter circuit (240 CNOT) and its 18-CNOT compression: loss on the grid, uniform certificate, design averages, Haar moments | Table of the certified compression |
| `verify_shots.py` | Number of device runs needed to confirm the compression loss | Shot counts after the compression table |
| `budget.py` | Query budgets with the value oracle and the classical Monte Carlo budget; intervals where the quantum budget is smaller | Appendix B, table of budgets |

`econst.py`, `testfun.py`, `compression_example.py` and `verify_shots.py` read
the group construction from `designs.py`; `verify_shots.py` also reads
`compression_example.py`. `compression_example.py` takes about one minute.

License: MIT (see `LICENSE`).
