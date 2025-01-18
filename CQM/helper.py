import random
import os
import json


def format_json_to_text(filepath):  # filepath is a json file
    weightCaps, weights, profits = parse_inputs(filepath)
    print(len(weights), len(weights[0]))
    for profitI in profits:
        print(" ".join(list(map(str, profitI))))
    for weightsI in weights:
        print(" ".join(list(map(str, weightsI))))
    print(" ".join(list(map(str, weightCaps))))


def parse_inputs(filepath):
    # Check file extension to determine format
    _, file_extension = os.path.splitext(filepath)

    if file_extension == ".json":
        # Parse JSON file
        with open(filepath, "r") as file:
            data = json.load(file)

        # Extract data from JSON
        M = data["numserv"]  # Number of agents
        N = data["numcli"]  # Number of tasks
        profits = [[num for num in row] for row in data["cost"]]  # Profits matrix
        weights = data["req"]  # Weights matrix
        weightCaps = data["cap"]  # Weight capacities

    elif file_extension == ".txt":
        # Parse plain text file
        with open(filepath, "r") as file:
            # Read the first line and split it to get M (agents) and N (tasks)
            M, N = map(int, file.readline().split())

            # Read the next M lines to get the profits (costs) for each agent-task pair
            profits = []
            for _ in range(M):
                profits.append(list(map(int, file.readline().split())))

            # Read the next M lines to get the weights (costs) for each agent-task pair
            weights = []
            for _ in range(M):
                weights.append(list(map(int, file.readline().split())))

            # Read the next line to get the weight capacities
            weightCaps = list(map(int, file.readline().split()))
    else:
        raise ValueError(
            "Unsupported file format. Please provide a .json or .txt file."
        )

    print("Data Loaded from File")
    return weightCaps, weights, profits


def random_inputs(M, N):
    # M is number of agents
    # N is number of tasks

    # Generate weight capacities for agents
    weightCaps = [random.randint(30, 50) for _ in range(M)]

    # Generate weights (costs for assigning tasks to agents)
    weights = [
        [random.randint(5, 20) for _ in range(N)]  # Random weights between 5 and 20
        for _ in range(M)
    ]

    # Generate profits (randomized for variability)
    profits = [
        [random.randint(50, 150) for _ in range(N)]  # Random profits between 50 and 150
        for _ in range(M)
    ]

    print("Data Generated")

    return weightCaps, weights, profits
