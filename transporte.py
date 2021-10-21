import sys
import numpy as np
import copy
from sympy import Symbol, linsolve
import pandas as pd

cost_matrix = []
allocation_matrix = []
indicators_matrix = []
supply = []
demand = []
u = 0
filename = ''

def get_arguments() -> str:
    """
    get the terminal arguments

    @return string
        return the method and the file gets from terminal
    """
    args = sys.argv
    if len(args) != 3:
        print("The arguments must be [method] [file]")
        exit(1)
    return args[1], args[2]

def parse_file(file: str) -> list:
    """
    parse the file into a list of lists

    @param: file: str   get the file
    @return list
        return the list of lists with the data of the file
    """
    f = open(file, "r")
    arr = []
    for line in f:
        data = line.replace('\n', "").split(",")
        arr.append(list(map(int, data)))
    f.close()
    return arr

def balance_costs() -> None:
    """
    balance the costs of the cost table
    append row of zeros or column of zeros

    @return None
    """
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
    """
    northwest corner algorithm
    get the northwest corner of the cost table and put the minimun value
    between supply and demand.

    @return None
    """
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

def get_diff_beetwen_mins_in_rows(rows_ignored, cols_ignored) -> list:
    """
    get the difference between the two minimun numbers of each row
    append the value to the list

    @param: rows_ignored:   list  ignore the rows used in vogel
    @param: column_ignored: list  ignore the columns used in vogel
    @return list
        return the list with the differences 
        between the two minimun numbers of each row
    """
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

def get_diff_beetwen_mins_in_cols(rows_ignored, cols_ignored) -> list:
    """
    get the difference between the two minimun numbers of each column
    append the value to the list

    @param: rows_ignored:   list  ignore the rows used in vogel
    @param: column_ignored: list  ignore the columns used in vogel
    @return list
        return the list with the differences 
        between the two minimun numbers of each column
    """
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

def get_max_of_differences(diff_in_rows, diff_in_cols) -> tuple:
    """
    get the max value of the differences in rows and columns

    @param: diff_in_rows:   list  differences of the two minimun values of the rows
    @param: diff_in_cols:   list  differences of the two minimun values of the columns
    @return tuple
        return the max value and bool flag
        to refer if it is from row or column
    """
    max_rows = np.nanmax(np.array(diff_in_rows))
    max_cols = np.nanmax(np.array(diff_in_cols))
    if max_rows > max_cols :
        return (diff_in_rows, True)
    elif max_rows < max_cols:
        return (diff_in_cols, False)
    else:
        return (diff_in_rows, True)

def get_min_index_of_row(row, cols_ignored) -> int:
    """
    get the min index of the row chosen

    @param: row:            list  row with the max value   
    @param: cols_ignored:   list  columns ignored in vogel
    @return int
        return an integer with the position of the minimun
        value of the row
    """
    new_row = []
    for i in range(len(row)):
        if i in cols_ignored:
            new_row.append(np.NaN)
        else:
            new_row.append(row[i])
    return new_row.index(np.nanmin(np.array(new_row)))

def get_min_index_of_col(col, rows_ignored) -> int:
    """
    get the min index of the column chosen

    @param: col:            list  column with the max value   
    @param: rows_ignored:   list  rows ignored in vogel
    @return int
        return an integer with the position of the minimun
        value of the column
    """
    new_col = []
    for i in range(len(col)):
        if i in rows_ignored:
            new_col.append(np.NaN)
        else:
            new_col.append(col[i])
    return new_col.index(np.nanmin(np.array(new_col)))
 
def vogel() -> None:
    """
    vogel algorithm
    get the difference between the two minimun numbers of each row and column
    get the max value of that substractions
    put the min(supply, demand) into the min value of the row or column chosen
    doing this operations until the supply and demand have been satisfied

    @return None
    """
    global u
    supply_copy = copy.deepcopy(supply)
    demand_copy = copy.deepcopy(demand)
    rows_ignored = []
    cols_ignored = []
    while supply_copy.count(0) != len(supply_copy) and \
          demand_copy.count(0) != len(demand_copy):
        diff_in_rows = get_diff_beetwen_mins_in_rows(rows_ignored, cols_ignored)
        diff_in_cols = get_diff_beetwen_mins_in_cols(rows_ignored, cols_ignored)
        tuple = get_max_of_differences(diff_in_rows, diff_in_cols) 
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
            min_col = get_min_index_of_col(cost_matrix[:, index_higher], rows_ignored)
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

def calculate_cost_indeces() -> list:
    """
    calculate the cost indices of russell's algorithm
    formula = max value of the row + max value of the col - actual position 

    @return list
        return a list with the ci values and their positions in tuple shape
    """
    cost_indices = []
    for i in range(len(cost_matrix)):
        max_cost_row = max(cost_matrix[i])
        for j in range(len(cost_matrix[0])):
            max_cost_col = max(cost_matrix[:,j])
            ic = max_cost_row + max_cost_col - cost_matrix[i][j]
            cost_indices.append([ic, (i,j)])
    return cost_indices

def get_cost_indices_by_priority() -> list:
    """
    sort the cost indices list

    @return list
        return a sorted list with the ci values and their positions in tuple shape
    """
    cost_indices = calculate_cost_indeces()
    cost_indices.sort(reverse=True)
    return cost_indices

def russell() -> None:
    """
    Russell's algorithm
    calculate the cost indices
    sort the ci by priority max -> min 
    put the min(supply, demand) until the supply and demand have been satisfied
    
    @return None
    """
    global u
    cost_indices_by_priority = get_cost_indices_by_priority()
    supply_copy = copy.deepcopy(supply)
    demand_copy = copy.deepcopy(demand)
    for i in range(len(cost_indices_by_priority)):
        if supply_copy.count(0) == len(supply_copy) and \
           demand_copy.count(0) == len(demand_copy):
           break
        supply_index = cost_indices_by_priority[i][1][0]
        demand_index = cost_indices_by_priority[i][1][1]
        if supply_copy[supply_index] == 0 or demand_copy[demand_index] == 0:
            continue
        else:
            if supply_copy[supply_index] == 0:
                minimun = demand_copy[demand_index]
            elif demand_copy[demand_index] == 0:
                minimun = supply_copy[supply_index]
            else:
                minimun = min(demand_copy[demand_index], supply_copy[supply_index])

            supply_copy[supply_index] -= minimun
            demand_copy[demand_index] -= minimun
            allocation_matrix[cost_indices_by_priority[i][1]] = minimun
            u += minimun * cost_matrix[cost_indices_by_priority[i][1]]

def get_u_headers() -> list:
    """
    create the list with the u headers
    [u1, u2, u3]
    
    @return list
    """
    global allocation_matrix
    u_headers = []
    for i in range(len(allocation_matrix)):
        u_headers.append(Symbol("u" + str(i)))
    return u_headers 

def get_v_headers() -> list:
    """
    create the list with the v headers
    [v1, v2, v3]
    
    @return list
    """
    global allocation_matrix
    v_headers = []
    for i in range(len(allocation_matrix[0])):
        v_headers.append(Symbol("v" + str(i)))
    return v_headers

def get_value_with_zero() -> list:
    """
    find the value with more allocations
    and replace it in the list with zero 
    [0, u2, u3]  
    
    @return list
    """
    u_headers = get_u_headers()
    v_headers = get_v_headers()
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
        u_headers[amount_alloc_in_us.index(max_u)] = 0
        return u_headers, v_headers
    elif max_u < max_v:
        v_headers[amount_alloc_in_vs.index(max_v)] = 0
        return u_headers, v_headers
    else:
        u_headers[amount_alloc_in_us.index(max_u)] = 0
        return u_headers, v_headers

def get_allocation_indices() -> list:
    """
    find the value with more allocations
    and replace it in the list with zero 
    [0, u2, u3]  
    
    @return list
    """
    global allocation_matrix
    allocation_indices = []
    for i in range(len(allocation_matrix)):
        for j in range(len(allocation_matrix[0])):
            if not np.isnan(allocation_matrix[i][j]):
                allocation_indices.append((i,j))
    return allocation_indices

def find_equations() -> list:
    """
    find the u solutions and the v solutions
    
    @return list
        return a list with the u solutions 
        and a list with the v solutions 
    """
    u_headers, v_headers = get_value_with_zero()
    allocation_indices = get_allocation_indices()
    equations = []
    for i, j in allocation_indices:
        u = u_headers[i]
        v = v_headers[j]
        c = cost_matrix[i][j]
        equation = u + v - c
        equations.append(equation)
    sol = linsolve(equations, (u_headers + v_headers)).args[0]
    amount_of_us = len(u_headers)
    u_sols = sol[:amount_of_us]
    v_sols = sol[amount_of_us:]
    return u_sols, v_sols

def fill_indicators_matrix() -> None:
    """
    fill the indicators table
    formula ui + vj - cij

    @return None
    """
    u_sols, v_sols = find_equations()
    for i in range(len(allocation_matrix)):
        for j in range(len(allocation_matrix[0])):
            if np.isnan(allocation_matrix[i][j]):
                equation = u_sols[i] + v_sols[j] - cost_matrix[i][j]
                indicators_matrix[i][j] = equation

def find_the_cycle() -> None:
    """
    Fill with nan's the rows and columns with an allocation
    until arrive to the incoming variable
    
    @return None
    """
    global indicators_matrix, allocation_matrix
    allocation_matrix_copy = copy.deepcopy(allocation_matrix)
    indicators_matrix = np.array(indicators_matrix)
    max_indicator_index = np.unravel_index(np.nanargmax(indicators_matrix, axis=None), indicators_matrix.shape)
    rows_visited = [max_indicator_index[0]]
    cols_visited = [max_indicator_index[1]]
    changes = True
    while changes:
        changes = False
        for i in range(len(allocation_matrix_copy)):
            if i in rows_visited:
                continue
            elif np.count_nonzero(~np.isnan(allocation_matrix_copy[i])) == 1:
                allocation_matrix_copy[i,:] = np.NaN
                changes = True
                break
        changes = False
        for j in range(len(allocation_matrix_copy[0])):
            if j in cols_visited:
                continue
            elif np.count_nonzero(~np.isnan(allocation_matrix_copy[:,j])) == 1:
                allocation_matrix_copy[:,j] = np.NaN
                changes = True
                break
    return allocation_matrix_copy, max_indicator_index

def change_variables() -> None:
    """
    create a list with the numbers of the cycle
    get the min value of the even indeces
    put that value in the incoming variable position
    substract that value to the other numbers
    update the allocation table

    @return None
    """
    allocation_matrix_copy, max_indicator_index = find_the_cycle()
    allocation_order = []
    max_indicator_copy = copy.deepcopy(max_indicator_index)
    non_Nan_elems_in_matrix =  np.count_nonzero(~np.isnan(allocation_matrix_copy))
    while True:

        for j in range(len(allocation_matrix_copy[0])):
            row_value = allocation_matrix_copy[max_indicator_copy[0]][j]  
            if not np.isnan(row_value):
                allocation_order.append([row_value, (max_indicator_copy[0], j)])
                max_indicator_copy = (max_indicator_copy[0], j)
                allocation_matrix_copy[max_indicator_copy[0]][j] = np.NaN
                break

        if len(allocation_order) == non_Nan_elems_in_matrix:
            break
        
        for i in range(len(allocation_matrix_copy)):
            col_value = allocation_matrix_copy[i][max_indicator_copy[1]]  
            if not np.isnan(col_value):
                allocation_order.append([col_value, (i, max_indicator_copy[1])])
                max_indicator_copy = (i, max_indicator_copy[1])
                allocation_matrix_copy[i][max_indicator_copy[1]] = np.NaN
                break

        if len(allocation_order) == non_Nan_elems_in_matrix:
            break

    allocation_order = np.array(allocation_order, dtype=object)
    evens = []
    for i in range(len(allocation_order)):
        if i % 2 == 0:
            evens.append(allocation_order[i][0])
    substaccion_value = min(evens)

    for i in range(len(allocation_order)):
        index = allocation_order[i][1]
        value = allocation_order[i][0]
        if i % 2 == 0:
            operation = value - substaccion_value
            if operation == 0:
                allocation_matrix[index] = np.NaN
            else:
                allocation_matrix[index] = value - substaccion_value
        else:
            allocation_matrix[index] = value + substaccion_value
    allocation_matrix[max_indicator_index] = substaccion_value

def get_transport_solution() -> None:
    """
    get the transport solution multiply 
    the allocations by the costs in their respective positions

    @return None
    """
    allocation_indices = get_allocation_indices()
    global u 
    u = 0
    for i,j in allocation_indices:
        u += cost_matrix[i,j] * allocation_matrix[i,j]

def is_optimal() -> bool:
    return np.nanmax(indicators_matrix) <= 0

def write_transport_solution(iteration: int) -> None:
    """
    write the transport solution in the file

    @param iteration: int 
    @return None
    """
    u_headers, v_headers = get_u_headers(), get_v_headers()
    u_headers = ["", *u_headers]
    v_headers = ["", *v_headers]
    u_sols, v_sols = find_equations()
    u_sols = [np.NaN, *u_sols]
    indicators_matrix_v = np.insert(indicators_matrix, 0, v_sols, axis=0)
    indicators_matrix_u_v = np.insert(indicators_matrix_v, 0, u_sols, axis=1)
    allocation_matrix_df = pd.DataFrame(data=allocation_matrix)
    indicators_matrix_df = pd.DataFrame(data=indicators_matrix_u_v, columns=v_headers, index=u_headers)
    new_file = '{0}_solution.txt'.format(filename)
    with open(new_file, "a") as f:
        txt_sol = f'Transport Algorithm\n\n' \
                  f'iteration: {iteration}\n\n' \
                  f'Indicators Table\n\n' \
                  f'{indicators_matrix_df}\n\n' \
                  f'Allocations Table\n\n' \
                  f'{allocation_matrix_df}\n\n' \
                  f'U: {u}\n\n'
        f.write(txt_sol)
    f.close

def transport() -> None:
    """
    this function use the above auxiliary functions
    to do the transport algorithm

    @return None
    """
    global indicators_matrix
    indicators_matrix = np.full((len(cost_matrix), len(cost_matrix[0])), np.NaN)
    i = 0
    while True:
        fill_indicators_matrix()
        if is_optimal():
            txt_sol = f'Transport Algorithm\n\n' \
                  f'[LAST ITERATION]\n' \
                  f'Allocations Table\n\n' \
                  f'{pd.DataFrame(data=allocation_matrix)}\n\n' \
                  f'U: {u}\n\n'
            print(txt_sol)
            break
        change_variables()
        get_transport_solution()
        write_transport_solution(i)
        i += 1
        indicators_matrix = np.full((len(cost_matrix), len(cost_matrix[0])), np.NaN)

def get_headers() -> list:
    """
    get the headers of the cost table
    [01, 02, 03]
    [D1, D2, D3]

    @return list
    """
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

def write_initial_solution(method: str) -> None:
    """
    write the initial solution in the file

    @param method: str   [northwest corner, vogel, russell's] 
    @return None
    """
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
        print(txt_sol)
    f.close

def is_degenerated() -> bool:
    """
    check if it is degenerated solution

    @return bool
    """
    allocation_indices = get_allocation_indices()
    m = len(allocation_matrix[0])
    n = len(allocation_matrix)
    return len(allocation_indices) != (m + n - 1)

if __name__ == "__main__":
    """
    main function

    @return None
    """
    method, file = get_arguments()
    filename = file.split('.')[0]
    data = parse_file(file)
    supply, demand = data[0], data[1]
    cost_matrix = np.array(data[2:])
    balance_costs()
    allocation_matrix = np.full((len(cost_matrix), len(cost_matrix[0])), np.NaN)
    if method == "1":
        north_west_corner()
        write_initial_solution("NORTHWEST CORNER")
        if is_degenerated():
            print("[DEGENERATED SOLUTION]")
            exit(1)
        transport()
    elif method == "2":
        vogel()
        write_initial_solution("VOGEL")
        if is_degenerated():
            print("[DEGENERATED SOLUTION]")
            exit(1)
        transport()
    elif method == "3":
        russell()
        write_initial_solution("RUSSELL")
        if is_degenerated():
            print("[DEGENERATED SOLUTION]")
            exit(1)
        transport()
