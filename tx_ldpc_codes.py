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

def estimate_ber(code, num_bits, EbN0_list, hard_decision=True, mpa=False):
    """ 
    BER simulation of a code.
    IN code: Object of the class LDPC_code
    IN EbN0_list: SNR range
    IN hard_decision: Flag. If True, hard-decision decoding is simulated, else soft-decision.
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
    code_bits = info_bits.dot(code.G) % 2
    # BPSK mapping.
    tx = (-1) ** np.array(code_bits)
    
    # Loop over Eb/N0 values.
    ber = np.empty(EbN0_list.size)
    for k, EbN0 in enumerate(EbN0_list):
        EsN0_lin = r * 10**(EbN0/10)
        # Apply AWGN Channel.
        rx = tx + np.random.randn(*tx.shape) / np.sqrt(2*EsN0_lin)
        if hard_decision:
            rx_bits = 0.5*(1-np.sign(rx)) # Hard decision to bits.
            epsilon = special.erfc(np.sqrt(EsN0_lin)) # Error probability of BSC.
            # Decode.
            llrs = np.array([code.decode_bsc(y, epsilon, spa_iters, mpa) for y in rx_bits])
        else: # Soft decision.
            # Decode.
            llrs = np.array([code.decode_awgn(y, EsN0_lin, spa_iters, mpa) for y in rx])
    
        # Determine BER.
        dec_bits = 0.5*(1-np.sign(llrs)) # Hard decision on LLRs.
        ber[k] = np.count_nonzero(np.logical_xor(code_bits, dec_bits)) / num_codebits
    return ber

# Load LDPC codes.
code1 = LDPC.LDPC_code(os.path.join(os.path.dirname(__file__),"WiGig1.alist"))
code2 = LDPC.LDPC_code(os.path.join(os.path.dirname(__file__),"WiGig2.alist"))

# Simulation parameters
num_bits = 100000
spa_iters = 5
EbN0_list = np.arange(2,7,0.5)

#ber1 = estimate_ber(code1, num_bits, EbN0_list)
#ber2 = estimate_ber(code2, num_bits, EbN0_list)
ber1 = estimate_ber(code1, num_bits, EbN0_list, hard_decision=False, mpa=True)
ber2 = estimate_ber(code2, num_bits, EbN0_list, hard_decision=False, mpa=True)
ber1_sd = estimate_ber(code1, num_bits, EbN0_list, hard_decision=False)
ber2_sd = estimate_ber(code2, num_bits, EbN0_list, hard_decision=False)

# Plot.
fig, ax = plt.subplots(num="Transmission Block Code")
ax.semilogy(EbN0_list, ber1)
ax.semilogy(EbN0_list, ber2)
ax.semilogy(EbN0_list, ber1_sd)
ax.semilogy(EbN0_list, ber2_sd)
plt.legend(['code1 mpa', 'code2 mpa', 'code1 spa', 'code2 spa'])

plt.xlabel(r"$E_\mathrm{b}/N_0$ (dB)", fontsize=16)
plt.ylabel("BER",fontsize=14)
plt.grid()
plt.show()
