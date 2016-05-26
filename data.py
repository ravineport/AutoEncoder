import numpy as np

def data_from_file(file_path):
    '''
    file_pathから訓練データを取得
    '''
    data = []
    for line in open(file_path, 'r'):
        data.append(list(map(float, line.split())))
    ans = {'N': int(data[0][0]), 'D': int(data[0][1]), 'data':np.array(data[1:])}
    return ans


def print_params2file(params, file_path):
    '''
    file_pathに対してparamsをスペース区切りで保存
    '''
    f = open(file_path,'w')

    for Wrow in params['W1']:
        f.write(' '.join(list(map(lambda x: str(x), Wrow))) + '\n')
    for Wrow in params['W2']:
        f.write(' '.join(list(map(lambda x: str(x), Wrow))) + '\n')
    f.write(' '.join(list(map(lambda x: str(x), params['b1']))) + '\n')
    f.write(' '.join(list(map(lambda x: str(x), params['b2']))) + '\n')
    f.close()
