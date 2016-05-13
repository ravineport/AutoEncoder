def outer(x, y):
    return [[xi*yj for yj in y] for xi in x]

if __name__ == '__main__':
    x = [1,2,3]
    y = [4,5,6,7]
    print(outer(x,y))
