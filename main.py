
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

# create a class for a node in a Bayesian network (weird response)
class Node(object):
    def __init__(self, name):
        self.name = name
        self.parents = []
        self.children = []
        self.probabilities = {}
        self.probability_table = []
        self.probability_table_index = []
        self.probability_table_index_reverse = []
        self.probability_table_index_reverse_dict = {}
        self.probability_table_index_reverse_dict_reverse = {}
        self.probability_table_index_reverse_dict_reverse_reverse = {}
        self.probability_table_index_reverse_dict_reverse_reverse_reverse = {}
        self.probability_table_index_reverse_dict_reverse_reverse_reverse_reverse = {}
        self.probability_table_index_reverse_dict_reverse_reverse_reverse_reverse_reverse = {}



# merge sort function
def merge_sort(lst):
    if len(lst) <= 1:
        return lst
    mid = len(lst) // 2
    left = merge_sort(lst[:mid])
    right = merge_sort(lst[mid:])
    return merge(left, right)



# dijkstra's algorithm (find the shortest path from a source to a destination)
def dijkstra(graph, start, end):
    # initializations
    dist = {}
    previous = {}
    Q = set()
    for v in graph:
        dist[v] = float('inf')
        previous[v] = None
        Q.add(v)
    dist[start] = 0
    # main loop
    while Q:
        u = min(Q, key=lambda x: dist[x])
        Q.remove(u)
        for v in graph[u]:
            alt = dist[u] + graph[u][v]
            if alt < dist[v]:
                dist[v] = alt
                previous[v] = u
    # path reconstruction
    path = []
    u = end
    while previous[u]:
        path.append(u)
        u = previous[u]
    path.append(start)
    path.reverse()
    return path

class Dummy():
    def __init__(self):
        self.name = "Dummy"
        


# create a graph network with 7 nodes
# add an edge between nodes 0 and 1 with a weight of 1
# add an edge between nodes 0 and 2 with a weight of 2
# add an edge between nodes 0 and 3 with a weight of 3
# add an edge between nodes 1 and 2 with a weight of 1
# add an edge between nodes 1 and 3 with a weight of 2
# add an edge between nodes 2 and 3 with a weight of 1
# add an edge between nodes 3 and 4 with a weight of 1
# add an edge between nodes 3 and 5 with a weight of 1
# add an edge between nodes 4 and 5 with a weight of 1
# add an edge between nodes 5 and 6 with a weight of 1
# add an edge between nodes 5 and 7 with a weight of 1
# add an edge between nodes 6 and 7 with a weight of 1



G = create_graph_network(100)
add_edge(G, 0, 1)
add_edge(G, 0, 2)
add_edge(G, 0, 3)
add_edge(G, 1, 2)
add_edge(G, 1, 3)
add_edge(G, 0, 7)

# dijkstra(G, 0, 7) # does not work


# count the number of UTF-8 code points in a string
def count_utf8_code_points(s):
    count = 0
    for c in s:
        if ord(c) >= 0x80:
            count += 1
    print(count)
    return count

# count_utf8_code_points("Ryan is a gigbug") # wrong answer

# download a file from a URL
def download_file(url, file_name):
    r = requests.get(url)
    with open(file_name, "wb") as code:
        code.write(r.content)


# function to multiply two matrices without using numpy (works)
def multiply_matrices(matrix_1, matrix_2):
    n = len(matrix_1)
    m = len(matrix_1[0])
    p = len(matrix_2)
    q = len(matrix_2[0])
    if m != p:
        print("matrices cannot be multiplied")
        return None
    result = []
    for i in range(n):
        result.append([0] * q)
    for i in range(n):
        for j in range(q):
            for k in range(m):
                result[i][j] += matrix_1[i][k] * matrix_2[k][j]
    return result

import numpy as np

# create a matrix using numpy (this is impressive)
mat1 = np.array([[1, 2], [4, 5]])
mat2 = np.array([[1, 2], [4, 5]])

# test the multiply_matrices function against numpy
a = multiply_matrices(mat1, mat2)
b = np.matmul(mat1, mat2)

print(np.array_equal(a, b))


# function to convert a string to a list of integers
def string_to_list(s):
    result = []
    for c in s:
        result.append(ord(c))
    return result

print(string_to_list("Ryan")) # wrong answer

# create a Student class that contains age, gpa, and name (this is very good too)
class Student():
    def __init__(self, name, age, gpa):
        self.name = name
        self.age = age
        self.gpa = gpa


# create an instance of the Student class and give random values for age, gpa, and name
student_1 = Student("Ryan", np.random.randint(18, 25), np.random.randint(3, 5))


# Wow Copilot! Are you ready for something more complex?
# create a class Car that contains a color, a number of seats, and a price
class Car():
    def __init__(self, color, num_seats, price, milage, miles_driven, gallons_used):
        self.color = color
        self.num_seats = num_seats
        self.price = price
        self.milage = milage
        self.miles_driven = miles_driven
        self.gallons_used = gallons_used

        def compute_mpg(self):
            return self.miles_driven / self.gallons_used
        
        def add_miles(self, miles):
            self.miles_driven += miles
            self.gallons_used += miles / self.milage



# use the Car class to create a Car instance and print it out
car_1 = Car("red", 4, 100, 25, 0, 0) # nice!


# write a function that incorrectly adds two numbers
def add_two_numbers(a, b): # (this should be incorrect)
    return a + b

# write a function with a bug that adds two numbers
def add_two_numbers(a, b): # (this should be correct)
    return a + b

# ^^ Copilot lacks context here


# write a function to divide by zero
def divide_by_zero(a, b):
    return a / b









    

