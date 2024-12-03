from dwave.system import LeapHybridCQMSampler
from dimod import ConstrainedQuadraticModel, BinaryQuadraticModel, QuadraticModel
import random


def parse_inputs(filename):
    N = 40
    # Weight capacities (limits for each agent)
    weightCaps = [random.randint(30, 50) for _ in range(N)]

    # Weights (costs for assigning tasks to agents)
    weights = [
        [random.randint(10, 25) for _ in range(N)]  # Random weights between 5 and 20
        for _ in range(N)
    ]

    # Profits (randomized to ensure variability and suitability for assignments)
    profits = [
        [random.randint(50, 150) for _ in range(N)]  # Random profits between 50 and 150
        for _ in range(N)
    ]

    return weightCaps, weights, profits


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
            sense="<=",
            rhs=1,
            label=f"assignment cap for task j = {j} ",
        )

    cqm.set_objective(obj)

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

    print(f"Total Profit : { totalProfit} ")


def main():

    sampler = LeapHybridCQMSampler()
    weightCaps, weights, profits = parse_inputs("testingInput")
    cqm = build_GAP_cqm(weightCaps, weights, profits)

    print("Submitting CQM to solver {}.".format(sampler.solver.name))
    sampleset = sampler.sample_cqm(cqm, label="GAP_CQM", time_limit=25)

    parse_solution(sampleset, weights, weightCaps, profits)


main()
