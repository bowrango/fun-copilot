
import random

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


