from dwave.system import DWaveSampler, EmbeddingComposite
from pyqubo import Binary
import numpy as np
from math import *
import sys

args = sys.argv[1:]
N = 20
w = np.array([7, 15, 11, 8, 7, 19, 11, 11, 4, 8, 3, 2, 12, 6, 2, 1, 12, 12, 17, 10])
p = np.array(
    [92, 60, 80, 15, 62, 62, 47, 62, 51, 55, 64, 3, 51, 7, 21, 73, 39, 18, 4, 89]
)
x = np.array([Binary(f"x{i}") for i in range(20)])
W = 50


def add_free_weights(W, x, w, p):
    binaries = ceil(log2(W))
    for i in range(binaries):
        x = np.append(x, Binary(f"x{len(x)}"))
        w = np.append(w, [2**i])
        p = np.append(p, 0)
    return [x, w, p]


x, w, p = add_free_weights(W, x, w, p)


A = 2 * np.sum(p)

hamiltonian = A * (W - (np.dot(w, x))) ** 2 - (np.dot(p, x))

# Compile the model into a QUBO
model = hamiltonian.compile()

# Generate the QUBO
qubo, offset = model.to_qubo()

# Print the QUBO
print("QUBO Matrix:", qubo)
print("Offset:", offset)

simulated_sampler = EmbeddingComposite(DWaveSampler())
response = simulated_sampler.sample_qubo(qubo, num_reads=int(args[0]))

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


for sample, energy in data[:10]:
    print(sample, energy, isValid(sample), getGain(sample))
