from dimod import SimulatedAnnealingSampler
from pyqubo import Binary
import numpy as np
from math import *

w = np.array([4, 5, 7, 12, 20, 12])
p0, p1, p2 = 3, 4, 6
p = np.array([3, 4, 6, 5, 50, 1])
x = np.array([Binary(f"x{i}") for i in range(len(w))])
W = 100


def add_free_weights(W, x, w, p):
    N = ceil(log2(W))
    for i in range(N):
        x = np.append(x, Binary(f"x{len(x)}"))
        w = np.append(w, [2**i])
        p = np.append(p, 0)
    return [x, w, p]


x, w, p = add_free_weights(W, x, w, p)

A = np.sum(p) + 1

hamiltonian = A * (W - (np.dot(w, x))) ** 2 - (np.dot(p, x))

# Compile the model into a QUBO
model = hamiltonian.compile()

# Generate the QUBO
qubo, offset = model.to_qubo()

# Print the QUBO
print("QUBO Matrix:", qubo)
print("Offset:", offset)

simulated_sampler = SimulatedAnnealingSampler()
response = simulated_sampler.sample_qubo(qubo, num_reads=100)

# Display the simulated annealing results
print("Simulated Solutions:")


data = [(sample, energy) for sample, energy in response.data(["sample", "energy"])]
sample, energy = data[0]
print(f"Sample: {sample}, Energy: {energy}")
