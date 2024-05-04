import numpy as np
from collections import deque

np.random.seed(34546)

dc_max = 5
dv_max = 5
n = 8
m = 4

def randomNodeDegrees(node_cnt, socket_cnt, d_max):
    '''
    radomly distributes sockets on nodes
    returns an sorted array with random degrees for @node_cnt nodes such that the degrees sumup to @socket_cnt
    while ensuring that: 1 <= degree <= d_max
    '''
    degrees = np.empty(node_cnt, dtype=int)
    available_sockets = socket_cnt
    for node_idx in range(node_cnt):
        min_degree = max(1, available_sockets - (node_cnt - node_idx - 1) * d_max)
        max_degree = min(d_max, available_sockets - (node_cnt - node_idx - 1))
        degrees[node_idx] = np.random.randint(min_degree, max_degree + 1, dtype=int)
        available_sockets -= degrees[node_idx]
    return np.sort(degrees)[::-1]

E = n + m - 1
varnode_sockets = randomNodeDegrees(n, E, dv_max)
checknode_sockets = randomNodeDegrees(m, E, dc_max)

H = np.zeros((m, n))
# bfs to populate H, ascending sorted arrays ensure minimal depth
q = deque()
q.append(0)
varnode_cnt = 1
checknode_cnt = 0
while len(q) > 0:
    varnode_idx = q[0]
    q.popleft()
    for checknode in range(varnode_sockets[varnode_idx] - (0 if varnode_idx == 0 else 1)):
        checknode_idx = checknode_cnt + checknode
        H[checknode_idx,varnode_idx] = 1
        q.extend(range(varnode_cnt, varnode_cnt + checknode_sockets[checknode_idx] - 1))
        H[checknode_idx,varnode_cnt:varnode_cnt + checknode_sockets[checknode_idx] - 1] = 1
        varnode_cnt += checknode_sockets[checknode_idx] - 1
    checknode_cnt += varnode_sockets[varnode_idx] - (0 if varnode_idx == 0 else 1)

print(H)


