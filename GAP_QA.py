from dwave.system import DWaveSampler, EmbeddingComposite
from pyqubo import Binary
import numpy as np
from math import *
import sys

args = sys.argv[1:]
import time

M, N = 3, 3  # Number of agents and tasks

# Weight matrix (cost of assigning task j to agent i)
w = [
    [3, 5, 7],
    [2, 4, 6],
    [1, 3, 5],
]

# Maximum allowable cost for each task
t = [10, 12, 8]

# Profit matrix (profit of assigning task j to agent i)
p = [
    [8, 10, 12],
    [7, 9, 11],
    [6, 8, 10],
]


def round_sig(num, sigs):
    # this is essentially e.g 0.4435 | sig 3 -> 443.5 and then back into decimal
    pow = -int(floor(log10(abs(num)))) + (sigs - 1)
    factor = 10**pow
    return round(num * factor) / factor


def createX():
    global M, N
    x = []
    for i in range(M):
        xj = []
        for j in range(N):
            xij = Binary(f"x{i}{j}")
            xj.append(xij)
        x.append(xj)
    return x


x = createX()


def add_free_weights():
    global M, N, t, x, w, p
    currentIndex = N

    for i in range(M):
        agent_budget_i = t[i]
        binaries = ceil(log2(agent_budget_i))
        for n in range(binaries):
            for y in range(M):
                x[y].append(Binary(f"x{y}{currentIndex}"))
                w[y].append(2**n)
                p[y].append(0)

            currentIndex += 1

    return [x, w, p]


x, w, p = add_free_weights()

A = np.sum(p) + 1


## Parts of hamiltonian
def gain():
    global x, p
    f = 0
    for i in range(len(x)):
        for j in range(len(x[0])):
            f += x[i][j] * p[i][j]
    return f


def budget():
    global x, w, t
    f = 0
    for i in range(len(x)):
        bi = 0
        for j in range(len(x[0])):
            bi += x[i][j] * w[i][j]
        bi = (bi - t[i]) ** 2
        f += bi
    return f


def assignment():
    global x
    f = 0
    for j in range(len(x[0])):
        a0 = sum([x[i][j] for i in range(len(x))])
        a1 = sum([x[i][j] for i in range(len(x))]) - 1
        f += a0 * a1
    return f


# print(gain(x, p))
# print(budget(x, w, t))
# print(assignment(x))

hamiltonian = -gain() + A * budget() + A * assignment()

# # Compile the model into a QUBO
model = hamiltonian.compile()

# # Generate the QUBO
qubo, offset = model.to_qubo()

# # Print the QUBO
print("QUBO Matrix:", qubo)
print("Offset:", offset)

sampler = EmbeddingComposite(DWaveSampler())
startTime = time.time()
response = sampler.sample_qubo(qubo, num_reads=int(args[0]))
endTime = time.time()
deltaTime = endTime - startTime


data = [(sample, energy) for sample, energy in response.data(["sample", "energy"])]


def checkBudget(sample):
    global x, w, t
    for i in range(len(x)):
        budget = t[i]
        tot = 0
        for j in range(len(x[0])):
            tot += sample[f"x{i}{j}"] * w[i][j]
        if budget < tot:
            return False
    return True


def checkAssignment(sample):
    global x
    for j in range(len(x[0])):
        tot = 0
        for i in range(len(x)):
            tot += sample[f"x{i}{j}"]
        if tot > 1:
            return False
    return True


def getGains(sample):
    global x, p
    tot = 0
    for i in range(len(x)):
        for j in range(len(x[0])):
            tot += sample[f"x{i}{j}"] * p[i][j]
    return tot


with open("GAP_QA_RESULTS.txt", "w") as f:
    original_stdout = sys.stdout
    sys.stdout = f

    print(f"Elapsed Time : {round_sig(deltaTime,3)}s\n")
    for i, (sample, energy) in enumerate(data):
        print(f"Sample {i} : {sample}")
        print(f"Energy : {energy}")
        print(f"Budget : { checkBudget(sample)}")
        print(f"Assignment : {checkAssignment(sample)}")
        print(f"Gains : {getGains(sample)}")
        print()
    sys.stdout = original_stdout
