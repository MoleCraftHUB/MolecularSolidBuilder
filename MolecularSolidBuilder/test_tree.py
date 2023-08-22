import numpy as np
import os,subprocess,sys,glob

def get_tree(S):
	n = len(S)
	L = set(range(1, n+2+1))
	tree_edges = []
	for i in range(n):
		u, v = S[0], min(L - set(S))
		S.pop(0)
		L.remove(v)
		tree_edges.append((u,v))
	tree_edges.append((L.pop(),L.pop()))
	return tree_edges



n = 10 # Kn with n vertices
N = 2 # generate 25 random trees with 20 vertices (as spanning trees of K20)
for i in range(N):
	S = np.random.choice(range(1,n+1), n-2, replace=True).tolist()
	T_E = get_tree(S) # the spanning tree corresponding to S
	print(T_E)
# plot the tree generated (with `networkx`, e.g.,)
