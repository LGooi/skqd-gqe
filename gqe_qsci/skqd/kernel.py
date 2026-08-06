import cudaq


@cudaq.kernel
def skqd_kernel(
    n_qubits: int,
    n_electrons: int,
    gqe_coeffs: list[float],
    gqe_words: list[cudaq.pauli_word],
    h_coeffs_scaled: list[float],
    h_words: list[cudaq.pauli_word],
    k: int,
    trotter_steps: int,
):
    q = cudaq.qvector(n_qubits)

    for i in range(n_electrons):
        x(q[i])

    for i in range(len(gqe_coeffs)):
        exp_pauli(gqe_coeffs[i], q, gqe_words[i])

    n_h_terms = len(h_coeffs_scaled)
    for s in range(k * trotter_steps):
        for j in range(n_h_terms):
            exp_pauli(h_coeffs_scaled[j], q, h_words[j])
