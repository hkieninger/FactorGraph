#!/usr/local/bin/python3
####################################################
# Code by Communications Engineering Lab (CEL), 2022
# Communication Lab - Chapter: Channel Coding
#
# Task: Transmission of LDPC codes.
####################################################
import numpy as np
from scipy import special
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({'axes.xmargin': 0})
import os
import LDPC

def estimate_ber(code, num_bits, EbN0_list, mpa=False):
    """ 
    BER simulation of a code.
    IN code: Object of the class LDPC_code
    IN EbN0_list: SNR range
    OUT: Estimated BER for each SNR value.
    """
    # Get code parameter.
    k = code.k
    n = code.n
    r = k/n
    num_cws = num_bits // k
    num_codebits = num_cws * n 
    
    # Generate random input bits.
    info_bits = np.random.randint(2, size=(num_cws, k))
    # Encoder
    code_bits = info_bits.dot(code.G) % 2 # shape (num_cws, n)
    # BPSK mapping.
    tx = (-1) ** np.array(code_bits)
    
    # Loop over Eb/N0 values.
    ber = np.empty(EbN0_list.size)
    fer = np.empty(EbN0_list.size)
    converged = np.empty(EbN0_list.size)
    for k, EbN0 in enumerate(EbN0_list):
        EsN0_lin = r * 10**(EbN0/10)
        # Apply AWGN Channel.
        rx = tx + np.random.randn(*tx.shape) / np.sqrt(2*EsN0_lin)
        llrs = np.empty(tx.shape)
        iters = np.empty(tx.shape[0])
        # Decode.
        for (i, y) in enumerate(rx):
            (llrs[i,:], iters[i]) = code.decode_awgn(y, EsN0_lin, spa_iters, max_product=mpa)
    
        # Determine BER.
        dec_bits = 0.5*(1-np.sign(llrs)) # Hard decision on LLRs.
        errors = np.logical_xor(code_bits, dec_bits)
        fer[k] = np.sum(np.count_nonzero(errors, axis=1) > 0) / num_cws
        ber[k] = np.count_nonzero(errors) / num_codebits
        converged_idx = (iters < spa_iters)
        converged_count = np.count_nonzero(converged_idx)
        converged[k] = converged_count / num_cws

    return (ber, fer, converged)

# Load LDPC codes.
code1 = LDPC.LDPC_code.fromAlistFile(os.path.join(os.path.dirname(__file__),"WiGig1.alist"))
code2 = LDPC.LDPC_code.fromAlistFile(os.path.join(os.path.dirname(__file__),"WiGig2.alist"))
code = code1

# Simulation parameters
num_bits = 100000
spa_iters = 20
EbN0_list = np.arange(0,4,0.5)

ber_mpa, fer_mpa, converged_mpa = estimate_ber(code, num_bits, EbN0_list, mpa=True)
ber_spa, fer_spa, converged_spa = estimate_ber(code, num_bits, EbN0_list)

# Plot.
fig, ax = plt.subplots(num="Transmission Block Code")
ax.semilogy(EbN0_list, ber_mpa)
ax.semilogy(EbN0_list, fer_mpa)
ax.semilogy(EbN0_list, converged_mpa)
ax.semilogy(EbN0_list, ber_spa)
ax.semilogy(EbN0_list, fer_spa)
ax.semilogy(EbN0_list, converged_spa)
plt.legend(['ber mpa', 'fer mpa', 'converged mpa', 'ber spa', 'fer spa', 'converged spa'])

plt.xlabel(r"$E_\mathrm{b}/N_0$ (dB)", fontsize=16)
plt.ylabel("BER",fontsize=14)
plt.grid()
plt.show()
