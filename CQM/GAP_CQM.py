from dwave.system import LeapHybridCQMSampler
from dimod import ConstrainedQuadraticModel, BinaryQuadraticModel, QuadraticModel
import sys
import os
import helper
import math
import time


def round_sig(num, sigs):
    # this is essentially e.g 0.4435 | sig 3 -> 443.5 and then back into decimal
    pow = -int(math.floor(math.log10(abs(num)))) + (sigs - 1)
    factor = 10**pow
    return round(num * factor) / factor


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
            obj.set_linear(index, -profit_ij)

            weight_constraint_i.add_variable("BINARY", index)
            weight_constraint_i.set_linear(index, weight_ij)
        cqm.add_constraint(
            weight_constraint_i,
            sense="<=",
            rhs=weightCap_i,
            label=f"weight cap for agent i = {i}",
        )

    for j in range(num_tasks):
        assignment_constraint_j = QuadraticModel()
        for i in range(num_agents):
            index = i * num_tasks + j
            assignment_constraint_j.add_variable("BINARY", index)
            assignment_constraint_j.set_linear(index, 1)
        cqm.add_constraint(
            assignment_constraint_j,
            sense="==",
            rhs=1,
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
    totalProfit = -best.energy

    print(f"Num Agents : {num_agents} | Num Tasks : {num_tasks}")
    print(f"Selected Item Coords : { selected_item_coords }")
    print(f"Number of Selected Items : { len(selected_item_coords) }")
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

    print(f"Total Profit : { -totalProfit} ")


def GAP_CQM_SOLVER(filepath, time_limit=10):
    sampler = LeapHybridCQMSampler()
    weightCaps, weights, profits = helper.parse_inputs(filepath)
    M, N = len(weights), len(weights[0])
    cqm = build_GAP_cqm(weightCaps, weights, profits)

    print("Submitting CQM to solver {}.".format(sampler.solver.name))

    startTime = time.time()
    sampleset = sampler.sample_cqm(
        cqm, label="GAP_CQM", time_limit=max(time_limit, int(0.00047 * M * N))
    )
    endTime = time.time()
    deltaTime = endTime - startTime

    print("Sampler Finished")

    result_file = open(f"results/{filepath.replace("/", ".")}.txt", "w")
    sys.stdout = result_file
    print(f"File Path : {filepath}")
    print(f"Time Elapsed : {round_sig(deltaTime, 3)}s")
    print("")

    parse_solution(sampleset, weights, weightCaps, profits)

    sys.stdout = sys.__stdout__
    result_file.close()

    print("Done!")


def GAP_CQM_RANDOM(M, N, time_limit=10):
    sampler = LeapHybridCQMSampler()
    weightCaps, weights, profits = helper.random_inputs(M, N)
    cqm = build_GAP_cqm(weightCaps, weights, profits)

    print("Submitting CQM to solver {}.".format(sampler.solver.name))
    sampleset = sampler.sample_cqm(cqm, label="GAP_CQM", time_limit=time_limit)

    parse_solution(sampleset, weights, weightCaps, profits)
