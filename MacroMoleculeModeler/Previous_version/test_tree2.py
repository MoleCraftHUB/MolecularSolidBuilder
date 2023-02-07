import os,sys,glob,subprocess
import numpy as np
import networkx as nx

tree = nx.random_tree(n=10, seed=0, create_using=nx.DiGraph)
print(tree)