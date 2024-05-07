import LDPC
import unittest

import numpy as np

def binaryArray(stop, bit_width):
    '''
    creates an array of binary numbers from 0 to stop-1
    returns a matrix of shape (stop, bit_width)
    [
    lsb_0, ..., msb_0;
        ...
    lsb_stop-1, ..., msb_stop-1;
    ]
    '''
    return ((np.arange(stop)[:, None] & (1 << np.arange(bit_width))) > 0).astype(int)

class TestLDPC(unittest.TestCase):

    num_cws = 100000
    # parity check matrix of code without cycles
    # H = np.array([[1, 0, 1, 0, 0],
    #               [0, 1, 1, 0, 1],
    #               [0, 0, 0, 1, 1]], dtype=int)
    H = np.load('codes/random_acyclic_LDPC.npy')
    
    # Hamming Code contains cycles -> should not work
    # H = np.array([
    #         [1, 0, 1, 0, 1, 0, 1],
    #         [0, 1, 1, 0, 0, 1, 1],
    #         [0, 0, 0, 1, 1, 1, 1]], dtype=int)

    EbN0 = 2
    spa_iters = 6

    def setUp(self):
        '''
        Generate self.rx: data y at ouput of AWGN channel
        self.rx.shape = (num_cws, n)
        '''
        self.code = LDPC.LDPC_code(self.H)
        (self.k, self.n)= self.code.G.shape
        self.codewords = (binaryArray(2**self.k, self.k) @ self.code.G) % 2 # shape (2**k, n)
        self.bpsk_cws = (-1)**self.codewords # codewords in bipolar format, shape (2**k, n)
        # Generate random input bits.
        info_bits = np.random.randint(2, size=(self.num_cws, self.k))
        # Encoder
        code_bits = (info_bits @ self.code.G) % 2
        # BPSK mapping.
        tx = (-1) ** code_bits
        # Apply AWGN Channel.
        r = self.k / self.n
        self.EsN0_lin =  r * 10**(self.EbN0/10)
        self.sigma = 1 / np.sqrt(2 * self.EsN0_lin)
        self.rx = tx + np.random.randn(*tx.shape) * self.sigma # shape (num_cws, n)

    def tearDown(self):
        pass

    def test_mpa(self):
        '''
        prueft ob das Ergebnis von MPA Blockwise-MAP entspricht, dies muss fuer einen Code ohne Zyklen erfuellt sein
        fuer den AWGN Kanal entspricht das MAP Codeword, dem Codewort mit der maximalen Korrelation mit dem Empfangenen y (siehe NT1 Zusammenfassung, Seite 12)
        Komplexitaet: O(2**k * n)

        TODO: account for tie (case if MAP has no single maximum)
        '''
        # compute Blockwise-MAP estimate
        correlations = self.rx @ self.bpsk_cws.T # shape (num_cws, 2**k)
        map_estimate = self.codewords[np.argmax(correlations, axis=1)] # shape (num_cws, n)

        # compute MPA assignment
        llrs = np.empty(self.rx.shape)
        iters = np.empty(self.rx.shape[0])
        for (i, y) in enumerate(self.rx):
            (llrs[i,:], iters[i]) = self.code.decode_awgn(y, self.EsN0_lin, self.spa_iters, max_product=True)
        mpa_assignment = 0.5*(1-np.sign(llrs)) # Hard decision on LLRs.

        # compare
        mpa_unequal_map = np.count_nonzero(np.logical_xor(map_estimate, mpa_assignment), axis=1) > 0 # shape (num_cws)
        self.assertEqual(np.count_nonzero(mpa_unequal_map), 0)

    def test_spa(self):
        '''
        prueft ob das Ergebnis von SPA Bitwise-MAP entspricht, dies muss fuer einen Code ohne Zyklen erfuellt sein
        argmax_xi P(xi | y) = argmax_xi sum_x P(xi, x | y) = argmax_xi sum_x P(y | xi, x) * P(xi | x) * P(x)
        = argmax_xi sum_x P(y | x) * 1{xi=(x)_i}

        P(xi = 1 | y) > P(xi = 0 | y)
        '''
        # compute Bitwise-MAP estimate
        y_minus_x_norm_squared = np.sum((self.rx[np.newaxis,:,:] - self.bpsk_cws[:,np.newaxis,:])**2, axis=2).T # shape (num_cws, 2**k)
        p_y_given_x = np.exp(-y_minus_x_norm_squared / (2 * self.sigma**2)) / (self.sigma * np.sqrt(2 * np.pi)) # shape (num_cws, 2**k)
        p_xi1_given_y = p_y_given_x @ self.codewords # proportional to p(xi = 1 | y) assuming uniform prior, shape (num_cws, n)
        p_xi0_given_y = p_y_given_x @ np.logical_not(self.codewords) # proportional to p(xi = 1 | y) assuming uniform prior, shape (num_cws, n)
        map_estimate = p_xi1_given_y > p_xi0_given_y # shape (num_cws, n)

        # compute SPA assignment
        llrs = np.empty(self.rx.shape)
        iters = np.empty(self.rx.shape[0])
        for (i, y) in enumerate(self.rx):
            (llrs[i,:], iters[i]) = self.code.decode_awgn(y, self.EsN0_lin, self.spa_iters, max_product=False)
        spa_assignment = llrs < 0 # Hard decision on LLRs.

        # compare
        mpa_unequal_map = np.count_nonzero(np.logical_xor(map_estimate, spa_assignment), axis=1) > 0 # shape (num_cws)
        self.assertEqual(np.count_nonzero(mpa_unequal_map), 0)