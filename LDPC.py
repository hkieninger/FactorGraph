#!/usr/local/bin/python3
####################################################
# Code by Communications Engineering Lab (CEL), 2022
# Communication Lab - Chapter: Channel Coding
####################################################
import numpy as np

class LDPC_code:
    """
    Class for irregular LDPC codes with
    non-optimized sum-product algorithm (SPA) decoder (belief propagation).
    """
    def __init__(self, parity_check_matrix):
        '''
        initialization
        IN: Parity check matrix in alist format.
        '''
        # Read H from alist file and set parameters of LDPC code.
        self.H = parity_check_matrix
        (self.m, self.n) = self.H.shape
        self.k = self.n - self.m
        self.dv_max = np.max(np.sum(self.H, axis=0)).astype(int) # Maximum variable node degree
        self.dc_max = np.max(np.sum(self.H, axis=1)).astype(int) # Maximum check node degree

        # Compute generator matrix G.
        self.G = self.make_gen()

        # Preparative computations which only need to be carried out once for a given H.
        (self.row, self.col, self.c_mask, self.c2v_reshape, self.v_mask, self.v2c_reshape) = self.prepare_decoder() 
        self.num_edges = len(self.row)
        self.ONE = 0.9999999999999 # For clipping before arctanh to avoid numerical instabilities.
        self.c2v_cview = np.empty((self.m, self.dc_max))
        self.c2v_vview = np.zeros((self.n, self.dv_max))

    @classmethod
    def fromAlistFile(cls, alist_filename):
        return cls(LDPC_code.read_alist_file(alist_filename))
        
    @staticmethod
    def read_alist_file(filename):
        """
        This function reads in an alist file and creates the
        corresponding parity check matrix H. The format of alist
        files is described at:
        http://www.inference.phy.cam.ac.uk/mackay/codes/alist.html
        """
        myfile    = open(filename,'r')
        data      = myfile.readlines()
        size      = str.split(data[0])
        numCols   = int(size[0])
        numRows   = int(size[1])
        H = np.zeros((numRows,numCols))
        for lineNumber in np.arange(4,4+numCols):
          line = np.fromstring(data[lineNumber], dtype=int, sep=' ')
          for index in line[line != 0]:
            H[int(index)-1,lineNumber-4] = 1
        return H

    def make_gen(self):
        """
        This function computes the corresponding generator matrix G to the given
        parity check matrix H.
        """
        A = self.H[:, : self.k]
        B = self.H[:, self.k :]
        F = A.T @ np.linalg.inv(B.T)
        G = np.concatenate((np.eye(self.k), F), axis=1)
        # Sanity check.
        assert not np.any((G @ self.H.T) % 2)
        return G

    def prepare_decoder(self):
        """
        For the SP algorithm, we use 2 different arrangements of the messages:
        -Check node view (cview): m rows, each containing up to dc_max messages from variable nodes
        -Variable node view (vview): n rows, each containing up to dv_max messages from check nodes
        
        This function calculates masks (in order to apply a computation only to valid entries) 
        and mappers (to switch between the 2 views).
        """
        (row, col) = np.nonzero(self.H)
        row_cview = row
        col_cview = np.concatenate([np.arange(i) for i in np.diff(np.flatnonzero(np.r_[True,row[:-1]!=row[1:],True]))])
        c_mask = np.zeros((self.m, self.dc_max), dtype=bool)
        c_mask[row_cview, col_cview] = True # Mask, which contains the valid entries of the check node view.

        c2v_reshape = np.argsort(col, kind='mergesort') # col_cview
        v2c_reshape = np.argsort(c2v_reshape, kind='mergesort')

        row_vview = np.sort(col)
        col_vview = np.concatenate([np.arange(i) for i in np.diff(np.flatnonzero(np.r_[True,row_vview[:-1]!=row_vview[1:],True]))])
        v_mask = np.zeros((self.n, self.dv_max), dtype=bool)
        v_mask[row_vview, col_vview] = True # Mask, which contains the valid entries of the variable node view.1
        return row, col, c_mask, c2v_reshape, v_mask, v2c_reshape

    def decode_bsc(self, rx_bits, epsilon, iters, max_product=False):
        """
        For given BSC(epsilon) channel, run 'iters' iterations of the BP algorithm.
        """
        # Channel parameter for LLR computation.
        Lc = np.log((1-epsilon)/epsilon)
        return self.spa((-1)**rx_bits * Lc, iters, max_product)
        
    def spa(self, input_LLR, iters, max_product=False):
        assert iters > 0

        one = float('inf') if max_product else 1

        # Initialization.
        v2c = np.ones((self.m, self.dc_max)) * one # previous v2c messages (channel LLR excluded)
        v2c[self.c_mask] = np.zeros(self.num_edges)
        v2c[self.c_mask] = input_LLR[self.col]

        previous_v2c = np.ones_like(v2c) * float('nan') # previous v2c messages to examine convergence

        for it in range(iters): # SPA iterations.
            # CN update.
            if max_product:
                for check_node_port in range(self.dc_max):
                    extrinsic = np.delete(v2c, check_node_port, axis=1)
                    self.c2v_cview[:,check_node_port] = np.prod(np.sign(extrinsic), axis=1) * np.min(np.abs(extrinsic), axis=1)
            else:
                v2c[self.c_mask] = np.tanh(v2c[self.c_mask]/2)
                for check_node_port in range(self.dc_max):
                    self.c2v_cview[:,check_node_port] = np.prod(np.delete(v2c, check_node_port, axis=1), axis=1)
                self.c2v_cview[self.c_mask] = 2*np.arctanh(np.clip(self.c2v_cview[self.c_mask], -self.ONE, self.ONE))
    
            # Reshape messages from check2variable nodes to variable view.
            self.c2v_vview[self.v_mask] = np.take_along_axis(self.c2v_cview[self.c_mask].flatten(), self.c2v_reshape, axis=0)
            
            # Total LLR.
            marginal = np.sum(self.c2v_vview, axis=1)
            output_LLR = marginal + input_LLR
            # VN Update.
            v2c[self.c_mask] = np.take_along_axis((output_LLR[:,None] - self.c2v_vview)[self.v_mask], self.v2c_reshape, axis=0)

            # check for convergence of messages
            # TODO: think about normalization for meaningful epsilon and mathematically clean convergence condition

        return (output_LLR, -1)
    
    def decode_awgn(self, rx, esn0_lin, iters, max_product=False):
        Lc = 4 * esn0_lin
        return self.spa(Lc * rx, iters, max_product)

