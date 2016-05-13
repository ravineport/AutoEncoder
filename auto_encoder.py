from operator import mul

def outer(x, y):
    return [[xi * yj for yj in y] for xi in x]


def affine(x, W, b):
    for wRow in W:
        assert len(wRow) == len(x)
    assert len(W) == len(b)

    h = [sum(map(mul, wRow, x)) for wRow in W]
    return [hi+bi for hi, bi in zip(h, b)]


if __name__ == '__main__':
    x = [1,2,3]
    y = [4,5,6,7]
    W = [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
    b = [13,14,15,16]
    print(outer(x, y))
    print(affine(x, W, b))
