from operator import mul, add, sub
import random
import pprint

def outer(x, y):
    return [[xi * yj for yj in y] for xi in x]


def affine(x, W, b):
    for wRow in W:
        assert len(wRow) == len(x)
    assert len(W) == len(b)

    h = [sum(map(mul, wRow, x)) for wRow in W]
    return list(map(add, h, b))


# matrix1 - matrix2
def subtract_M2_from_M1(matrix1, matrix2):
    for row1, row2 in zip(matrix1, matrix2):
        assert len(row1) == len(row2)
    assert len(matrix1) == len(matrix2)

    return [list(map(sub, M1row, M2row)) for M1row, M2row in zip(matrix1, matrix2)]


# a * matrix
def multiply_matrix_by_a(matrix, a):
    return [[a * element  for element in row] for row in matrix]


def auto_encoder(file_path, hidden_dim_num=5, epoch_num=1000):
    data = data_from_file(file_path)
    elements_num = data['N']
    params = initialize_params_randomly(hidden_dim_num, data['D'])

    for i in range(epoch_num):
        total_loss = 0.0
        for x in data['data']:
            params['x'] = x
            params_new = learn_one_iteration(**params)
            params.update(params_new)
            total_loss += params['loss']
            del params['loss']
        average_loss = total_loss / elements_num
        params['eta'] = params['eta']*0.9
        print(i, average_loss)

    return params


def learn_one_iteration(x, W1, W2, b1, b2, eta):
    args = {'x': x, 'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}
    args_forward = args.copy()
    args_bp = args.copy()
    args['eta'] = eta
    del args['x']

    forward_result = forward(**args_forward)
    # pprint.pprint(forward_result)
    loss = forward_result['loss']
    del forward_result['loss']
    args_bp.update(forward_result)

    bp_result = back_propagation(**args_bp)
    # pprint.pprint(bp_result)
    args.update(bp_result)

    param_result = param_update(**args)
    param_result['loss'] = loss
    return param_result


def forward(x, W1, W2, b1, b2):
    h = affine(x, W1, b1)
    y = affine(h, W2, b2)
    assert len(x) == len(y)
    loss = 1.0/2.0 * sum(map(lambda e: e**2, map(sub, x, y)))
    return {'h': h, 'y': y, 'loss': loss}


def back_propagation(x, h, y, W1, W2, b1, b2):
    gy = list(map(sub, y, x))
    gW2 = outer(gy, h)
    W2T = list(map(list, zip(*W2)))
    # for wRow in W2T:
    #     assert len(wRow) == len(gy)
    gh = [sum(map(mul, wRow, gy)) for wRow in W2T]
    gW1 = outer(gh, x)
    return {'gW1': gW1, 'gW2': gW2, 'gb1': gh, 'gb2': gy}


def param_update(W1, W2, b1, b2, gW1, gW2, gb1, gb2, eta):
    etaTimesW1 = multiply_matrix_by_a(gW1, eta)
    W1new = subtract_M2_from_M1(W1, etaTimesW1)
    etaTimesW2 = multiply_matrix_by_a(gW2, eta)
    W2new = subtract_M2_from_M1(W2, etaTimesW2)
    b1new = list(map(sub, b1, list(map(lambda x: eta * x, gb1))))
    b2new = list(map(sub, b2, list(map(lambda x: eta * x, gb2))))
    return {'W1': W1new, 'W2': W2new, 'b1': b1new, 'b2': b2new}


def data_from_file(file_path):
    data = []
    for line in open(file_path, 'r'):
        data.append(list(map(float, line.split())))
    ans = {'N': int(data[0][0]), 'D': int(data[0][1]), 'data':data[1:]}
    return ans


def initialize_params_randomly(hidden_dim_num, data_dim_num):
    W1 = []
    W2 = []
    b1 = [0.0] * hidden_dim_num
    b2 = [0.0] * data_dim_num
    for i in range(hidden_dim_num):
        W1row = []
        for j in range(data_dim_num):
            W1row.append(random.uniform(-0.05, 0.05))
            #W1row.append(random.random())
        W1.append(W1row)

    for i in range(data_dim_num):
        W2row = []
        for j in range(hidden_dim_num):
            W2row.append(random.uniform(-0.05, 0.05))
            #W2row.append(random.random())
        W2.append(W2row)

    return {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2, 'eta': 0.01}


def initialize_eta():
    pass


if __name__ == '__main__':
    # x = [1,2,3]
    # y = [4,5,6,7,8]
    # W1 = [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
    # W2 = [[1,2,3,4,5], [4,5,6,7,8], [7,8,9,10,11], [10,11,12,13,14]]
    # b1 = [13,14,15,16]
    # b2 = [1,2,3,4,5]
    # eta = 0.01
    x = [1,2,3,4,5]
    y = [1,2,3,4,5]
    W1 = [[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]]
    W2 = [[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15]]
    b1 = [1,2,3]
    b2 = [1,2,3,4,5]
    eta = 0.01
    # pprint.pprint(learn_one_iteration(x, W1, W2, b1, b2, eta))
    # print(outer(x, y))
    # print(affine(x, W1, b1))
    #print(learn_one_iteration(x, W1, W2, b1, b2, eta))
    #print(data_from_file('./data/dataset.dat'))
    #print(initialize_params_randomly(5, 10))
    print(auto_encoder('./data/dataset.dat'))
