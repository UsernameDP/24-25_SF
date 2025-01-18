from dwave.system import LeapHybridCQMSampler
from neal import SimulatedAnnealingSampler
from dimod import ConstrainedQuadraticModel, BinaryQuadraticModel, QuadraticModel
import sys
import os
import helper
from math import *
import math
import time
import matplotlib.pyplot as plt


def round_sig(num, sigs):
    # this is essentially e.g 0.4435 | sig 3 -> 443.5 and then back into decimal
    pow = -int(math.floor(math.log10(abs(num)))) + (sigs - 1)
    factor = 10**pow
    return round(num * factor) / factor


def meanPriority(weights, profits):
    num_agents = len(weights)
    num_tasks = len(weights[0])

    tot = 0

    for i in range(num_agents):
        for j in range(num_tasks):
            tot += profits[i][j] / weights[i][j]

    return tot / (num_agents * num_tasks)


def getMeanProfit(profits):
    num_agents = len(profits)
    num_tasks = len(profits[0])

    tot = 0

    for i in range(num_agents):
        for j in range(num_tasks):
            tot += profits[i][j]
    return tot / (num_agents * num_tasks)


def build_GAP_cqm(weightCaps, weights, profits):

    num_agents = len(weights)
    num_tasks = len(weights[0])

    cqm = ConstrainedQuadraticModel()
    obj = BinaryQuadraticModel(vartype="BINARY")

    for i in range(num_agents):
        weight_constraint_i = QuadraticModel()
        weightCap_i = weightCaps[i]
        weights_i = weights[i]
        profits_i = profits[i]

        for j in range(num_tasks):
            profit_ij = profits_i[j]
            weight_ij = weights_i[j]

            index = i * num_tasks + j
            obj.add_variable(index)
            obj.set_linear(index, profit_ij)

            weight_constraint_i.add_variable("BINARY", index)
            weight_constraint_i.set_linear(index, weight_ij)

        cqm.add_constraint(
            weight_constraint_i <= weightCap_i, label=f"weight cap for agent i = {i}"
        )

    for j in range(num_tasks):
        assignment_constraint_j = QuadraticModel()
        for i in range(num_agents):
            index = i * num_tasks + j
            assignment_constraint_j.add_variable("BINARY", index)
            assignment_constraint_j.set_linear(index, 1)
        cqm.add_constraint(
            assignment_constraint_j == 1,
            label=f"assignment cap for task j = {j} ",
        )

    cqm.set_objective(obj)

    print("CQM Generated")

    return cqm


def parse_solution(sampleset, weights, weightCaps, profits):
    feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)

    if not len(feasible_sampleset):
        raise ValueError("No feasible solution found")

    best = feasible_sampleset.first

    num_agents = len(weights)
    num_tasks = len(weights[0])

    selected_item_indices = [key for key, val in best.sample.items() if val == 1.0]
    selected_item_coords = [
        (indice // num_tasks, indice % num_tasks) for indice in selected_item_indices
    ]
    selected_weights = [[] for i in range(num_agents)]
    selected_profits = [[] for i in range(num_agents)]
    for r, c in selected_item_coords:
        selected_weights[r].append(weights[r][c])
        selected_profits[r].append(profits[r][c])
    selected_weights_totals = [
        sum(selected_weight) for selected_weight in selected_weights
    ]
    selected_profits_gained = [
        sum(selected_profit) for selected_profit in selected_profits
    ]

    print(f"Num Agents : {num_agents} | Num Tasks : {num_tasks}")
    print()
    print("Weigths : ")
    for i in range(num_agents):
        print(f"Agent {i} : { weights[i] }")
    print()
    print("Profits : ")
    for i in range(num_agents):
        print(f"Agent {i} : { profits[i] }")
    print()
    print(f"Selected Item Coords : { selected_item_coords }")
    print()
    print(f"Number of Selected Items : { len(selected_item_coords) }")
    print()
    print(f"Selected Weights : ")
    for i in range(len(selected_weights)):
        print(
            f"Agent {i} : { selected_weights[i] } total weight : {selected_weights_totals[i]} cap : {weightCaps[i]}"
        )

    print()
    print(f"Selected Profits : ")
    for i in range(len(selected_profits)):
        print(
            f"Agent {i} : { selected_profits[i] } profit : {selected_profits_gained[i]}"
        )
    totalProfit = 0
    for profits_gained_i in selected_profits_gained:
        totalProfit += profits_gained_i

    print()
    print(f"Total Profit : { totalProfit} ")
    print(f"Energy : {best.energy}")

    sys.stdout = sys.__stdout__

    selected_weights_1d = sum(selected_weights, [])
    selected_profits_1d = sum(selected_profits, [])

    weights_1d = sum(weights, [])
    profits_1d = sum(profits, [])

    # plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st plot
    # plt.scatter(selected_weights_1d, selected_profits_1d, s=50, alpha=0.2)
    # plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd plot
    # plt.scatter(weights_1d, profits_1d, c="gray", s=75, alpha=0.2, label="Not Chosen")
    # plt.show()


def GAP_CQM_SOLVER(filepath):
    sampler = LeapHybridCQMSampler()
    weightCaps, weights, profits = helper.parse_inputs(filepath)
    M, N = len(weights), len(weights[0])
    cqm = build_GAP_cqm(weightCaps, weights, profits)

    print("Submitting CQM to solver {}.".format(sampler.solver.name))
    print(cqm)

    sampleset = sampler.sample_cqm(
        cqm,
        label="GAP_CQM",
        time_limit=LeapHybridCQMSampler.min_time_limit(sampler, cqm) * 2,
    )

    result_file = open(f"results/{filepath.replace("/", ".")}.txt", "w")
    sys.stdout = result_file
    print(f"File Path : {filepath}")
    print(f"Run Time : {sampleset.info["run_time"] / 1000000}s")

    parse_solution(sampleset, weights, weightCaps, profits)

    result_file.close()

    print("Done!")


def GAP_CQM_RANDOM(M, N, time_limit=10):
    sampler = LeapHybridCQMSampler()
    weightCaps, weights, profits = helper.random_inputs(M, N)
    cqm = build_GAP_cqm(weightCaps, weights, profits)

    print("Submitting CQM to solver {}.".format(sampler.solver.name))
    sampleset = sampler.sample_cqm(cqm, label="GAP_CQM", time_limit=time_limit)

    parse_solution(sampleset, weights, weightCaps, profits)
