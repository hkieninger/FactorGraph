import numpy as np
from collections import deque

#np.random.seed(34546)

dc_max = 5
dv_max = 5
n = 16
m = 8

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

'''
Um den längsten Pfad in einem Baum zu finden können wir einfach zweimal BFS anwenden.
(siehe: https://www.geeksforgeeks.org/longest-path-undirected-tree/)
Wir nutzen den BFS durchlauf auch direkt um sicher zu stellen, dass der Graph keinen Zyklus enthält.
(siehe: https://soi.ch/wiki/bfs/)
'''

def bfs(H_tree, root):
    '''
    @H_tree: parity-check matrix of a Tree Code
    @root: root node for BFS, (0 for var node / 1 for check node, node idx)
    @return: list of depths of the nodes
    '''
    q = deque()
    q.append(root)

    (m, n) = H_tree.shape

    depths = [n * [-1], m * [-1]]
    depths[root[0]][root[1]] = 0

    parents = [n * [None], m * [None]]

    current_vertex = None
    while len(q) > 0:
        current_vertex = q[0]
        q.popleft()
        H = H_tree if current_vertex[0] == 0 else H_tree.T 
        for node in range(H.shape[0]):
            if H[node,current_vertex[1]] and parents[current_vertex[0]][current_vertex[1]] != (not current_vertex[0],node):
                assert(depths[not current_vertex[0]][node] < 0)
                depths[not current_vertex[0]][node] = depths[current_vertex[0]][current_vertex[1]] + 1
                parents[not current_vertex[0]][node] = current_vertex
                q.append((not current_vertex[0], node))
    return depths, current_vertex

def longestPath(H_tree):
    depths, end_node1 = bfs(H_tree, (0, 0))
    # assert that we have a tree and not a forest
    for node_list in depths:
        for depth in node_list:
            assert(depth >= 0)
    
    depths, end_node2 = bfs(H_tree, end_node1)
    return depths[end_node2[0]][end_node2[1]]

print(longestPath(H))


