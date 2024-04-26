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
    converged = np.empty(EbN0_list.size)
    ber_converged = np.empty(EbN0_list.size)
    for k, EbN0 in enumerate(EbN0_list):
        EsN0_lin = r * 10**(EbN0/10)
        # Apply AWGN Channel.
        rx = tx + np.random.randn(*tx.shape) / np.sqrt(2*EsN0_lin)
        llrs = np.empty(tx.shape)
        iters = np.empty(tx.shape[0])
        if hard_decision:
            rx_bits = 0.5*(1-np.sign(rx)) # Hard decision to bits.
            epsilon = special.erfc(np.sqrt(EsN0_lin)) # Error probability of BSC.
            # Decode.
            for (i, y) in enumerate(rx_bits):
                (llrs[i,:], iters[i]) = code.decode_bsc(y, epsilon, spa_iters, mpa)
        else: # Soft decision.
            # Decode.
            for (i, y) in enumerate(rx):
                (llrs[i,:], iters[i]) = code.decode_awgn(y, EsN0_lin, spa_iters, mpa)
    
        # Determine BER.
        dec_bits = 0.5*(1-np.sign(llrs)) # Hard decision on LLRs.
        ber[k] = np.count_nonzero(np.logical_xor(code_bits, dec_bits)) / num_codebits
        converged_idx = (iters != -1)
        converged_count = np.count_nonzero(converged_idx)
        converged[k] = converged_count / num_cws
        ber_converged[k] = np.count_nonzero(np.logical_xor(code_bits[converged_idx,:], dec_bits[converged_idx,:])) / (converged_count * n) 

    return (ber, converged, ber_converged)

# Load LDPC codes.
code1 = LDPC.LDPC_code.fromAlistFile(os.path.join(os.path.dirname(__file__),"WiGig1.alist"))
code2 = LDPC.LDPC_code.fromAlistFile(os.path.join(os.path.dirname(__file__),"WiGig2.alist"))

# Simulation parameters
num_bits = 100000
spa_iters = 5
EbN0_list = np.arange(2,7,0.5)

#ber1 = estimate_ber(code1, num_bits, EbN0_list)
#ber2 = estimate_ber(code2, num_bits, EbN0_list)
ber1_mpa, converged_mpa, ber_converged_mpa = estimate_ber(code1, num_bits, EbN0_list, hard_decision=False, mpa=True)
#ber2_mpa = estimate_ber(code2, num_bits, EbN0_list, hard_decision=False, mpa=True)
ber1_spa, converged_spa, ber_converged_spa = estimate_ber(code1, num_bits, EbN0_list, hard_decision=False)
#ber2_spa = estimate_ber(code2, num_bits, EbN0_list, hard_decision=False)

# Plot.
fig, ax = plt.subplots(num="Transmission Block Code")
ax.semilogy(EbN0_list, ber1_mpa)
ax.semilogy(EbN0_list, converged_mpa)
ax.semilogy(EbN0_list, ber_converged_mpa)
#ax.semilogy(EbN0_list, ber2_mpa)
ax.semilogy(EbN0_list, ber1_spa)
ax.semilogy(EbN0_list, converged_spa)
ax.semilogy(EbN0_list, ber_converged_spa)
#ax.semilogy(EbN0_list, ber2_spa)
plt.legend(['code1 mpa', 'converged mpa', 'ber converged mpa', 'code1 spa', 'converged spa', 'ber converged spa'])

plt.xlabel(r"$E_\mathrm{b}/N_0$ (dB)", fontsize=16)
plt.ylabel("BER",fontsize=14)
plt.grid()
plt.show()
