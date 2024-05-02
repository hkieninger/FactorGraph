import numpy as np
import LDPC
from test_LDPC import binaryArray
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({'axes.xmargin': 0})

num_cws = 3
# parity check matrix of code without cycles
H_tree = np.array([
    [1, 0, 1, 0, 0],
    [0, 1, 1, 0, 1],
    [0, 0, 0, 1, 1]], dtype=int)
H_hamming= np.array([
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]], dtype=int)
EbN0 = -5
spa_iters = 20


code = LDPC.LDPC_code(H_hamming)
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

output_LLRs = np.empty((num_cws, spa_iters, n))
for (i, y) in enumerate(rx):
    print('new codeword')
    llr_generator = code.decode_awgn(y, EsN0_lin, spa_iters, max_product=True)
    for (j, iter) in enumerate(llr_generator):
        output_LLRs[i, j, :] = iter

bits = 3
fig, axes = plt.subplots(bits, 1)
fig.suptitle('A tale of 2 subplots')
for bit in range(bits):
    axes[bit].plot(output_LLRs[0,:,bit])
    axes[bit].set_ylabel(f'bit {bit}')

plt.show()