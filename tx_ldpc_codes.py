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
#mpl.rcParams.update({'axes.xmargin': 0})
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
    ber_converged = np.empty(EbN0_list.size)
    fer_converged = np.empty(EbN0_list.size)
    ber_converged_valid = np.empty(EbN0_list.size)
    fer_converged_valid= np.empty(EbN0_list.size)
    converged = np.empty(EbN0_list.size)
    converged_valid = np.empty(EbN0_list.size)
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

        ber_converged[k] = float('nan') if converged_count == 0 else np.count_nonzero(errors[converged_idx,:]) / (converged_count * n)
        fer_converged[k] = float('nan') if converged_count == 0 else np.sum(np.count_nonzero(errors[converged_idx,:], axis=1) > 0) / converged_count

        converged_valid_cw_idx = np.logical_and(converged_idx, np.count_nonzero((dec_bits @ code.H.T) % 2, axis=1) == 0)
        converged_valid_cw_cnt = np.count_nonzero(converged_valid_cw_idx)
        converged_valid[k] = converged_valid_cw_cnt / num_cws

        ber_converged_valid[k] = float('nan') if converged_valid_cw_cnt == 0 else np.count_nonzero(errors[converged_valid_cw_idx,:]) / (converged_valid_cw_cnt * n)
        fer_converged_valid[k] = float('nan') if converged_valid_cw_cnt == 0 else np.sum(np.count_nonzero(errors[converged_valid_cw_idx,:], axis=1) > 0) / converged_valid_cw_cnt

    return (ber, fer, converged, ber_converged, fer_converged, converged_valid, ber_converged_valid, fer_converged_valid)

with np.errstate(invalid="raise"):
    # Load LDPC codes.
    code1 = LDPC.LDPC_code.fromAlistFile(os.path.join(os.path.dirname(__file__),"codes/WiGig1.alist"))
    code2 = LDPC.LDPC_code.fromAlistFile(os.path.join(os.path.dirname(__file__),"codes/WiGig2.alist"))

    H_hamming= np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]], dtype=int)
    # H_tree = np.array([
    #     [1, 0, 1, 0, 0],
    #     [0, 1, 1, 0, 1],
    #     [0, 0, 0, 1, 1]], dtype=int)
    H_tree = np.load('codes/random_acyclic_LDPC.npy')
    bch_63_45 = np.load('codes/BCH_63_45.npz')['H']
    istc_39_24 = np.load('codes/ISTC39_24_opt_Nils.npz')['Hopt']
    golay_24_12 = np.load('codes/opt_golay24_12_with_ends_even_more_weights.npz')['Hopt']

    code3 = LDPC.LDPC_code(H_tree)

    code = code3

    # Simulation parameters
    num_bits = 100000
    spa_iters = 10
    EbN0_list = np.arange(-2,5,0.5)

    ber_mpa, fer_mpa, converged_mpa, ber_converged_mpa, fer_converged_mpa, converged_valid_mpa, ber_converged_valid_mpa, fer_converged_valid_mpa = estimate_ber(code, num_bits, EbN0_list, mpa=True)
    ber_spa, fer_spa, converged_spa, ber_converged_spa, fer_converged_spa, converged_valid_spa, ber_converged_valid_spa, fer_converged_valid_spa = estimate_ber(code, num_bits, EbN0_list)

    # Plot.
    fig, axes = plt.subplots(3, 1)

    axes[0].set_title('BER')
    axes[0].semilogy(EbN0_list, ber_mpa, label='mpa', linestyle='-', linewidth=4)
    axes[0].semilogy(EbN0_list, ber_spa, label='spa', linestyle='--', linewidth=3.5)
    axes[0].semilogy(EbN0_list, ber_converged_mpa, label='mpa converged', linestyle='-.', linewidth=3)
    axes[0].semilogy(EbN0_list, ber_converged_spa, label='spa converged', linestyle=':', linewidth=2.5)
    axes[0].grid()
    axes[0].legend()

    axes[1].set_title('FER')
    axes[1].semilogy(EbN0_list, fer_mpa, label='mpa', linestyle='-', linewidth=4)
    axes[1].semilogy(EbN0_list, fer_spa, label='spa', linestyle='--', linewidth=3.5)
    axes[1].semilogy(EbN0_list, fer_converged_mpa, label='mpa converged', linestyle='-.', linewidth=3)
    axes[1].semilogy(EbN0_list, fer_converged_spa, label='spa converged', linestyle=':', linewidth=2.5)
    axes[1].grid()
    axes[1].legend()

    axes[2].set_title('Convergence')
    axes[2].plot(EbN0_list, converged_mpa, label='mpa', linestyle='-', linewidth=4)
    print(f'converged mpa: {converged_mpa}')
    axes[2].plot(EbN0_list, converged_spa, label='spa', linestyle='--', linewidth=3.5)
    print(f'converged spa: {converged_spa}')
    axes[2].plot(EbN0_list, converged_valid_mpa, label='mpa valid', linestyle='-.', linewidth=3)
    print(f'converged valid mpa: {converged_valid_mpa}')
    axes[2].plot(EbN0_list, converged_valid_spa, label='spa valid', linestyle=':', linewidth=2.5)
    print(f'converged valid spa: {converged_valid_spa}')
    axes[2].grid()
    axes[2].legend()

    # axes[3].plot(EbN0_list, ber_converged_valid_mpa, label='mpa ber converged valid')
    # axes[3].plot(EbN0_list, ber_converged_valid_spa, label='spa ber converged valid')
    # axes[3].plot(EbN0_list, fer_converged_valid_mpa, label='mpa fer converged valid')
    # axes[3].plot(EbN0_list, fer_converged_valid_spa, label='spa fer converged valid')
    # axes[3].grid()
    # axes[3].legend()

    plt.xlabel(r"$E_\mathrm{b}/N_0$ (dB)", fontsize=16)
    plt.show()

    pass
