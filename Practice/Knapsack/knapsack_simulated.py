from dimod import SimulatedAnnealingSampler
from pyqubo import Binary
import numpy as np
from math import *

N = 7
w = np.array([4, 5, 6, 12, 15, 55, 1])
p = np.array([3, 4, 10, 12, 6, 10000, 24])
x = np.array([Binary(f"x{i}") for i in range(N)])
W = 24


def add_free_weights(W, x, w, p):
    binaries = ceil(log2(W))
    for i in range(binaries):
        x = np.append(x, Binary(f"x{len(x)}"))
        w = np.append(w, [2**i])
        p = np.append(p, 0)
    return [x, w, p]


x, w, p = add_free_weights(W, x, w, p)
print(x, w, p)

A = 2 * np.sum(p)

hamiltonian = A * (W - (np.dot(w, x))) ** 2 - (np.dot(p, x))
print(hamiltonian)

# Compile the model into a QUBO
model = hamiltonian.compile()

# Generate the QUBO
qubo, offset = model.to_qubo()

# Print the QUBO
print("QUBO Matrix:", qubo)
print("Offset:", offset)

simulated_sampler = SimulatedAnnealingSampler()
response = simulated_sampler.sample_qubo(qubo, num_reads=125)

# Display the simulated annealing results
print("Simulated Solutions:")


data = [(sample, energy) for sample, energy in response.data(["sample", "energy"])]
sample, energy = data[0]


def isValid(sample):
    global x, w, p, N
    weight = 0
    for i in range(N):
        xi = f"x{i}"
        weight += w[i] * sample[xi]

    check_weight = 0
    for i in range(len(w)):
        xi = f"x{i}"
        check_weight += w[i] * sample[xi]

    if W >= weight and check_weight == W:
        return True
    return False


def getGain(sample):
    global x, w, p, N
    sum = 0
    for i in range(N):
        xi = f"x{i}"
        sum += p[i] * sample[xi]
    return sum


for sample, energy in data:
    print(sample, energy, isValid(sample), getGain(sample))
