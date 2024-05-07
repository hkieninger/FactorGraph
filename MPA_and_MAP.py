import numpy as np
import LDPC
from test_LDPC import binaryArray

# Hamming-Code
H = np.array([
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]], dtype=int)
golay_24_12 = np.load('codes/opt_golay24_12_with_ends_even_more_weights.npz')['Hopt']

num_cws = 10000
EbN0 = 2
spa_iters = 30

'''
setUp
'''
code = LDPC.LDPC_code(golay_24_12)
(k, n)= code.G.shape
codewords = (binaryArray(2**k, k) @ code.G) % 2 # shape (2**k, n)
bpsk_cws = (-1)**codewords # codewords in bipolar format, shape (2**k, n)
# Generate random input bits.
info_bits = np.random.randint(2, size=(num_cws, k))
# Encoder
code_bits = (info_bits @ code.G) % 2
# BPSK mapping.
tx = (-1) ** code_bits
# Apply AWGN Channel.
r = k / n
EsN0_lin =  r * 10**(EbN0/10)
sigma = 1 / np.sqrt(2 * EsN0_lin)
rx = tx + np.random.randn(*tx.shape) * sigma # shape (num_cws, n)

'''
check if MPA after convergence gives MAP
'''
# compute Blockwise-MAP estimate
correlations = rx @ bpsk_cws.T # shape (num_cws, 2**k)
map_estimate = codewords[np.argmax(correlations, axis=1)] # shape (num_cws, n)

# compute MPA assignment
llrs = np.empty(rx.shape)
iters = np.empty(rx.shape[0])
for (i, y) in enumerate(rx):
    (llrs[i,:], iters[i]) = code.decode_awgn(y, EsN0_lin, spa_iters, max_product=True)
mpa_assignment = 0.5*(1-np.sign(llrs)) # Hard decision on LLRs.

# compare converged MPA assignments with MAP
converged_idx = (iters < spa_iters)
mpa_unequal_map = np.count_nonzero(np.logical_xor(map_estimate[converged_idx,:], mpa_assignment[converged_idx,:]), axis=1) > 0 # shape (num_cws)
print(f'converged_count: {np.sum(converged_idx)}, difference between MPA and MAP: {np.count_nonzero(mpa_unequal_map)}')

# negative test, compare aswell non converged MAP assignments
mpa_unequal_map = np.count_nonzero(np.logical_xor(map_estimate, mpa_assignment), axis=1) > 0 # shape (num_cws)
print(f'converged & nonconverged: difference between MPA and MAP: {np.count_nonzero(mpa_unequal_map)}')