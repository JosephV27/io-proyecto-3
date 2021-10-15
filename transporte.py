import sys
import numpy as np
import copy
from sympy import Symbol, linsolve
import pandas as pd
from collections import Counter

cost_matrix = []
allocation_matrix = []
indicators_matrix = []
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
        cost_matrix = np.r_[cost_matrix, [row]]
    elif difference > 0:
        demand.append(abs(difference))
        # append a column
        column = np.full(len(cost_matrix), 0)
        cost_matrix = np.c_[cost_matrix, column]
    
def north_west_corner() -> None:
    global u
    supply_copy = copy.deepcopy(supply)
    demand_copy = copy.deepcopy(demand)
    i = 0
    j = 0
    bfs = []
    while supply_copy.count(0) != len(supply_copy) and \
          demand_copy.count(0) != len(demand_copy):
        minimum = min(supply_copy[i], demand_copy[j])
        supply_copy[i] -= minimum
        demand_copy[j] -= minimum
        allocation_matrix[i][j] = minimum
        u += cost_matrix[i][j] * minimum
        if supply_copy[i] == 0 and i < len(supply) - 1:
            i += 1
        if demand_copy[j] == 0 and j < len(demand) - 1:
            j += 1

# take the diff beetwen the two min numbers of each row and column 
def get_diff_beetwen_mins_in_rows(rows_ignored, cols_ignored):
    global cost_matrix
    diff_in_rows = []
    for i in range(len(cost_matrix)):
        row = []
        if i in rows_ignored:
            diff_in_rows.append(np.NaN)
        else:
            for j in range(len(cost_matrix[0])):
                if j in cols_ignored:
                    continue
                row.append(cost_matrix[i][j])
        sorted = np.sort(row)
        if len(sorted) == 0:
            continue
        elif len(sorted) == 1:
            diff_in_rows.append(sorted[0])
        else: 
            mins = sorted[:2]
            diff_in_rows.append(mins[1] - mins[0])
    return diff_in_rows

def get_diff_beetwen_mins_in_cols(rows_ignored, cols_ignored):
    global cost_matrix
    diff_in_cols = []
    for i in range(len(cost_matrix[0])):
        col = []
        if i in cols_ignored:
            diff_in_cols.append(np.NaN)
        else:
            for j in range(len(cost_matrix)):
                if j in rows_ignored:
                    continue
                col.append(cost_matrix[j][i])
        sorted = np.sort(col)
        if len(sorted) == 0:
            continue
        elif len(sorted) == 1:
            diff_in_cols.append(sorted[0])
        else: 
            mins = sorted[:2]
            diff_in_cols.append(mins[1] - mins[0])
    return diff_in_cols

def get_max(diff_in_rows, diff_in_cols):
    max_rows = np.nanmax(np.array(diff_in_rows))
    max_cols = np.nanmax(np.array(diff_in_cols))
    if max_rows > max_cols :
        return (diff_in_rows, True)
    elif max_rows < max_cols:
        return (diff_in_cols, False)
    else:
        return (diff_in_rows, True)

def get_min_index_of_row(row, cols_ignored):
    new_row = []
    for i in range(len(row)):
        if i in cols_ignored:
            new_row.append(np.NaN)
        else:
            new_row.append(row[i])
    return new_row.index(np.nanmin(np.array(new_row)))

def get_min_index_of_col(col, rows_ignored):
    new_col = []
    for i in range(len(col)):
        if i in rows_ignored:
            new_col.append(np.NaN)
        else:
            new_col.append(col[i])
    return new_col.index(np.nanmin(np.array(new_col)))
 

def vogel() -> None:
    global u
    supply_copy = copy.deepcopy(supply)
    demand_copy = copy.deepcopy(demand)
    rows_ignored = []
    cols_ignored = []
    while supply_copy.count(0) != len(supply_copy) and \
          demand_copy.count(0) != len(demand_copy):
        diff_in_rows = get_diff_beetwen_mins_in_rows(rows_ignored, cols_ignored)
        diff_in_cols = get_diff_beetwen_mins_in_cols(rows_ignored, cols_ignored)
        tuple = get_max(diff_in_rows, diff_in_cols) 
        flag = tuple[1]
        if flag:
            index_higher = diff_in_rows.index(np.nanmax(np.array(tuple[0])))
            # get the min of the row
            min_row = get_min_index_of_row(cost_matrix[index_higher], cols_ignored)
            if demand_copy[min_row] <= 0:
                minimum = supply_copy[index_higher]
            elif supply_copy[index_higher] <= 0:
                minimum = demand_copy[min_row]
            else:
                minimum = min(supply_copy[index_higher], demand_copy[min_row])
            allocation_matrix[index_higher][min_row] = minimum
            u += cost_matrix[index_higher][min_row] * minimum
            demand_copy[min_row] -= minimum
            supply_copy[index_higher] -= minimum
            if demand_copy[min_row] == 0:
                # insert columns finished
                cols_ignored.append(min_row)
            elif supply_copy[index_higher] == 0:
                # insert rows finished
                rows_ignored.append(index_higher)

        else:
            index_higher = diff_in_cols.index(np.nanmax(np.array(tuple[0])))
            # get the min of the col
            min_col = get_min_index_of_col(cost_matrix[:, index_higher], cols_ignored)
            if demand_copy[index_higher] <= 0:
                minimum = supply_copy[min_col]
            elif supply_copy[min_col] <= 0:
                minimum = demand_copy[index_higher]
            else:
                minimum = min(supply_copy[min_col], demand_copy[index_higher])
            allocation_matrix[min_col][index_higher] = minimum
            u += cost_matrix[min_col][index_higher] * minimum
            demand_copy[index_higher] -= minimum
            supply_copy[min_col] -= minimum
            if demand_copy[index_higher] == 0:
                # insert columns finished
                cols_ignored.append(index_higher)
            elif supply_copy[min_col] == 0:
                # insert rows finished
                rows_ignored.append(min_col)

def get_u_list() -> list:
    global allocation_matrix
    u_list = []
    for i in range(len(allocation_matrix)):
        u_list.append(Symbol("u" + str(i)))
    return u_list

def get_v_list() -> list:
    global allocation_matrix
    v_list = []
    for i in range(len(allocation_matrix[0])):
        v_list.append(Symbol("v" + str(i)))
    return v_list

def get_value_with_zero() -> str:
    # find the value with more allocations
    u_list = get_u_list()
    v_list = get_v_list()
    amount_alloc_in_us = []
    amount_alloc_in_vs = []
    for i in range(len(allocation_matrix)):
        amount_allocations = len(allocation_matrix[i]) - np.count_nonzero(np.isnan(allocation_matrix[i])) 
        amount_alloc_in_us.append(amount_allocations)
    for j in range(len(allocation_matrix[0])):
        amount_allocations = len(allocation_matrix[:,j]) - np.count_nonzero(np.isnan(allocation_matrix[:, j])) 
        amount_alloc_in_vs.append(amount_allocations)
    max_u = max(amount_alloc_in_us)
    max_v = max(amount_alloc_in_vs)
    if max_u > max_v:
        u_list[amount_alloc_in_us.index(max_u)] = 0
        return u_list, v_list
    elif max_u < max_v:
        v_list[amount_alloc_in_vs.index(max_v)] = 0
        return u_list, v_list
    else:
        u_list[amount_alloc_in_us.index(max_u)] = 0
        return u_list, v_list

def get_allocation_indices():
    global allocation_matrix
    allocation_indices = []
    print(allocation_matrix)
    for i in range(len(allocation_matrix)):
        for j in range(len(allocation_matrix[0])):
            if not np.isnan(allocation_matrix[i][j]):
                allocation_indices.append((i,j))
    return allocation_indices

def find_equations():
    u_list, v_list = get_value_with_zero()
    allocation_indices = get_allocation_indices()
    equations = []
    for i, j in allocation_indices:
        u = u_list[i]
        v = v_list[j]
        c = cost_matrix[i][j]
        equation = u + v - c
        equations.append(equation)
    sol = linsolve(equations, (u_list + v_list)).args[0]
    amount_of_us = len(u_list)
    sol_u = sol[:amount_of_us]
    sol_v = sol[amount_of_us:]
    return sol_u, sol_v

def fill_indicators_matrix():
    sol_u, sol_v = find_equations()
    for i in range(len(allocation_matrix)):
        for j in range(len(allocation_matrix[0])):
            if np.isnan(allocation_matrix[i][j]):
                equation = sol_u[i] + sol_v[j] - cost_matrix[i][j]
                indicators_matrix[i][j] = equation


def find_the_cycle():
    global indicators_matrix, allocation_matrix
    allocation_matrix_copy = copy.deepcopy(allocation_matrix)
    indicators_matrix = np.array(indicators_matrix)
    max_indicator_index = np.unravel_index(np.nanargmax(indicators_matrix, axis=None), indicators_matrix.shape)
    print(max_indicator_index)
    rows_visited = [max_indicator_index[0]]
    cols_visited = [max_indicator_index[1]]
    flag = True
    while flag:
        flag = False
        for i in range(len(allocation_matrix_copy)):
            if i in rows_visited:
                continue
            elif np.count_nonzero(~np.isnan(allocation_matrix_copy[i])) == 1:
                allocation_matrix_copy[i,:] = np.NaN
                flag = True
                break
        flag = False
        for j in range(len(allocation_matrix_copy[0])):
            if j in cols_visited:
                continue
            elif np.count_nonzero(~np.isnan(allocation_matrix_copy[:,j])) == 1:
                allocation_matrix_copy[:,j] = np.NaN
                flag = True
                break
    print()
    print(allocation_matrix_copy)

# def transport():
#     zero_value = get_value_with_zero()
#     print(zero_value)

def get_headers() -> list:
    global cost_matrix
    rows_header = []
    cols_header = []
    for i in range(len(cost_matrix)):
        rows_header.append("O" + str(i))
    rows_header.append("Demand")
    for j in range(len(cost_matrix[0])):
        cols_header.append("D" + str(j))
    cols_header.append("Supply")
    return rows_header, cols_header

def write_initial_solution(filename, method):
    rows_header, cols_header = get_headers()
    supply_copy = copy.deepcopy(supply)
    supply_copy.append(np.NaN)
    cost_matrix_copy = np.r_[cost_matrix, [demand]]
    cost_matrix_copy = np.c_[cost_matrix_copy, supply_copy]
    cost_matrix_df = pd.DataFrame(data=cost_matrix_copy, columns=cols_header, index=rows_header)
    allocation_matrix_df = pd.DataFrame(data=allocation_matrix)
    new_file = '{0}_solution.txt'.format(filename)
    with open(new_file, "a") as f:
        txt_sol = f'PROBLEM {filename}\n\n' \
                  f'{cost_matrix_df}\n\n' \
                  f'Initial Solution [{method}]\n\n' \
                  f'{allocation_matrix_df}\n\n' \
                  f'U: {u}\n\n'
        f.write(txt_sol)
    f.close


if __name__ == "__main__":
    method, file = get_arguments()
    filename = file.split('.')[0]
    data = parse_file(file)
    supply, demand = data[0], data[1]
    cost_matrix = np.array(data[2:])
    balance_costs()
    allocation_matrix = np.full((len(cost_matrix), len(cost_matrix[0])), np.NaN)
    indicators_matrix = np.full((len(cost_matrix), len(cost_matrix[0])), np.NaN)
    vogel()
    # write_initial_solution(filename, "VOGEL")
    fill_indicators_matrix()
    find_the_cycle()

