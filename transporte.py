import sys
import numpy as np
from numpy.lib.shape_base import column_stack
from numpy.matrixlib import matrix

cost_matrix = []
allocation_matrix = []
supply = []
demand = []
u = 0

def get_arguments() -> str:
    args = sys.argv
    if len(args) != 3:
        print("The arguments must be [method] [file]")
        # exit(1)
    return args[1], args[2]

def parse_file(file: str) -> list:
    f = open(file, "r")
    arr = []
    for line in f:
        data = line.replace('\n', "").split(",")
        arr.append(list(map(int, data)))
    f.close()
    return arr

def balance_costs() -> None:
    global cost_matrix
    difference = sum(supply) - sum(demand)
    if difference < 0:
        supply.append(abs(difference))
        # append a row
        row = np.full(len(cost_matrix[0]), 0)
        cost_matrix = np.r_[cost_matrix, row]
    elif difference > 0:
        demand.append(abs(difference))
        # append a column
        column = np.full(len(cost_matrix), 0)
        cost_matrix = np.c_[cost_matrix, column]
    
def north_west_corner() -> None:
    global u
    supply_copy = supply
    demand_copy = demand
    i = 0
    j = 0
    bfs = []
    while supply_copy.count(supply_copy[0]) != len(supply_copy) and \
          demand_copy.count(demand_copy[0]) != len(demand_copy):
        minimum = min(supply_copy[i], demand_copy[j])
        supply_copy[i] -= minimum
        demand_copy[j] -= minimum
        # bfs.append(((i, j), minimum))
        allocation_matrix[i][j] = minimum
        u += cost_matrix[i][j] * minimum
        if supply_copy[i] == 0 and i < len(supply) - 1:
            i += 1
        if demand_copy[j] == 0 and j < len(demand) - 1:
            j += 1
    # return bfs

def vogel() -> None:
    return 0

if __name__ == "__main__":
    method, file = get_arguments()
    data = parse_file(file)
    supply, demand = data[0], data[1]
    cost_matrix = np.array(data[2:])
    balance_costs()
    allocation_matrix = np.full((len(cost_matrix), len(cost_matrix[0])), np.NaN)
    # print(cost_matrix)
    north_west_corner()
    print(allocation_matrix)
    print(u)
