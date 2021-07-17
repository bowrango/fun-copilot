
import random
import numpy as np
from math import pi, exp

# function that adds two numbers
function = lambda x, y: x + y
print(function(2, 3))

# function to create an empty graph network
def create_graph_network(n):
    graph = {}
    for i in range(n):
        graph[i] = []

    return graph

# function to add an edge to the graph network (it actually generated this comment)
def add_edge(graph, v1, v2):
    graph[v1].append(v2)
    graph[v2].append(v1)
    return graph

# perform a random walk on the graph network (used the random module, which wasnt imported)
def random_walk(graph, start, max_steps):
    walk = [start]
    for i in range(max_steps):
        cur = walk[-1]
        if cur not in graph:
            break
        options = graph[cur]
        walk.append(random.choice(options))
    return walk

# create feature embedding for nodes in a graph network (this is crazy!)
def feature_embedding(graph, nodes):
    features = {}
    for n in nodes:
        walk = random_walk(graph, n, 100) # (implemented its own random walk function) <- it actually predicted exactly what I was going to write 
        d = dict.fromkeys(walk, 0)
        for i in range(len(walk)):
            d[walk[i]] = i
        features[n] = d
    return features

# create a Bayesian network (this is not quite right )
def create_bayesian_network(n):
    graph = create_graph_network(n)
    for i in range(n):
        for j in range(i+1, n):
            add_edge(graph, i, j)
    return graph

# function to get the probability of a given path (this is also not quite right)
def get_probability(path, features):
    d = {}
    for i in range(len(path)-1):
        d[path[i]] = features[path[i+1]]
    d[path[-1]] = features[path[0]]
    return d

# Wow Github Copilot, you are so smart!

# function to get all factors of a number (used reduce?)
def factors(n):
    return set(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

# function to get the eigenvalues of a matrix without numpy 
def eigenvalues_manual(matrix):
    n = len(matrix)
    eigvals = []
    for i in range(n):
        for j in range(i+1, n):
            matrix[i][j] = matrix[j][i] = -matrix[i][j]
    for i in range(n):
        matrix[i][i] = 0
        for j in range(n):
            matrix[j][i] = 0
    for i in range(n):
        eigvals.append(sum(matrix[i]))
    return eigvals

# function to get eigvenvectors of a matrix with numpy
def eigenvectors_manual(matrix):
    n = len(matrix)
    eigvals, eigvecs = np.linalg.eig(matrix) # lol, this was my first attempt at numpy
    return eigvals, eigvecs

# function to compute fft of signal 
def fft(signal):
    n = len(signal)
    if n == 1:
        return signal
    even = fft(signal[0::2])
    odd = fft(signal[1::2])
    T = [exp(-2j*pi*k/n)*odd[k] for k in range(n//2)]
    return [even[k] + T[k] for k in range(n//2)] + [even[k] - T[k] for k in range(n//2)]


