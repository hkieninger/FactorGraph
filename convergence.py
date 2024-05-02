import numpy as np
import LDPC
from test_LDPC import binaryArray
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({'axes.xmargin': 0})

'''
parameters
'''
np.random.seed(784538)

rows = 4
columns = 4
# parity check matrix of code without cycles
H_tree = np.array([
    [1, 0, 1, 0, 0],
    [0, 1, 1, 0, 1],
    [0, 0, 0, 1, 1]], dtype=int)
H_hamming= np.array([
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]], dtype=int)
H = H_hamming
EbN0 = -5
spa_iters = 30

'''
setup code and test data
'''
code = LDPC.LDPC_code(H)
(k, n)= code.G.shape
codewords = (binaryArray(2**k, k) @ code.G) % 2 # shape (2**k, n)
bpsk_cws = (-1)**codewords # codewords in bipolar format, shape (2**k, n)
# Generate random input bits.
num_cws = rows * columns
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
compute max absolute difference between successive messages
'''
epsilons = np.empty((num_cws, spa_iters-1))
for (cw_idx, y) in enumerate(rx):
    generator = code.decode_awgn(y, EsN0_lin, spa_iters, max_product=False)
    for (iter_idx, epsilon) in enumerate(generator):
        epsilons[cw_idx, iter_idx] = epsilon

'''
plot and display the results
'''
fig, axes = plt.subplots(rows, columns)
fig.suptitle(r'max_i(|m_k - m_{k-1}|)')
for r in range(rows):
    for c in range(columns):
        axes[r,c].plot(epsilons[r*columns+c,:])
        axes[r,c].set_ylim(ymin=0)
plt.show()