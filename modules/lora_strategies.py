def constant(ratio):
    def inner(depth):
        return ratio
    return inner
def poly(*args):
    xys = sorted(args)
    xs = [x for x, y in xys]
    ys = [y for x, y in xys]
    def inner(x):
        if(x<=xs[0]):
            return ys[0]
        if(x>=xs[-1]):
            return ys[-1]
        for idx, xy0 in enumerate(xys):
            x0, y0 = xy0
            x1, y1 = xys[idx+1]
            if(x0<=x and x<=x1):
                ratio = (x-x0)/(x1-x0)
                return y0*(1-ratio) + y1*ratio
        assert False
    return inner

def get_fstrategy(x):
    return eval(x)