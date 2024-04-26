'''
Klasse um den spezialfall zu untersuchen der ungefähr mit wahrscheinlichkeit 10**-4 auftritt
es kann passieren, dass der Decoder noch nicht konvergiert ist, aber in einem Zwischenschritt ein gültiges Codewort erreicht ist
in der LDPC.py Version von Luca wird dann das Codewort aus dem Zwischenschritt geliefert
'''

import numpy as np
import LDPC
from test_LDPC import binaryArray

def gauss(x, mu, sigma):
    return np.exp(-((x - mu)/sigma)**2/2) / (sigma * np.sqrt(2 * np.pi))

spa_iters = 5
#rx = np.array([-1.34970704,  0.47403465,  1.13036284,  1.18454389, -0.57350856])
rx = np.array([-1.3,  0.5,  1.1,  1.2, -0.6])
H = np.array([[1, 0, 1, 0, 0],
              [0, 1, 1, 0, 1],
              [0, 0, 0, 1, 1]], dtype=int)
code = LDPC.LDPC_code(H)
EbN0 = 2
r = code.k / code.n
EsN0_lin = r * 10**(EbN0/10)
sigma = 1 / np.sqrt(2 * EsN0_lin)

p_y_given_x_eq_0 = gauss(rx, (-1)**0, sigma)

# compute MAP estimate
codewords = (binaryArray(2**code.k, code.k) @ code.G) % 2 # shape (2**k, n)
bpsk_cws = (-1)**codewords # shape (2**k, n)
correlations = rx @ bpsk_cws.T # shape (num_cws, 2**k)
map_estimate = codewords[np.argmax(correlations)] 
print(map_estimate)

# compute MPA assignment
(llrs, iters) = code.decode_awgn(rx, EsN0_lin, spa_iters, max_product=True)
mpa_assignment = 0.5*(1-np.sign(llrs))
print(mpa_assignment)

