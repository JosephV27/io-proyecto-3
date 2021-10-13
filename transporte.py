from os import linesep
import sys
import numpy as np
from numpy.core.fromnumeric import sort
import copy

from numpy.lib.nanfunctions import nanmin

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
        # bfs.append(((i, j), minimum))
        allocation_matrix[i][j] = minimum
        u += cost_matrix[i][j] * minimum
        if supply_copy[i] == 0 and i < len(supply) - 1:
            i += 1
        if demand_copy[j] == 0 and j < len(demand) - 1:
            j += 1
    # return bfs'


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

if __name__ == "__main__":
    method, file = get_arguments()
    data = parse_file(file)
    supply, demand = data[0], data[1]
    cost_matrix = np.array(data[2:])
    balance_costs()
    allocation_matrix = np.full((len(cost_matrix), len(cost_matrix[0])), np.NaN)
    vogel()
    print(allocation_matrix)
    print(u)
    # print(cost_matrix)
    # north_west_corner()
    # print(allocation_matrix)
    # print(u)
