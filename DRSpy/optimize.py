from DRSpy.analysis import np

####### Func

def w_avg(x, w):
    """
    Weighted Average
    """
    x, w = np.array(x), np.array(w)
    return  (x*w).sum()/w.sum()

def quenching(x, a, c, *args):
    pass

####### Fit

def landau_fit():
    pass
def moyal_fit():
    pass

def linear_fit(x, a, b):
    return a*x+b

def gauss():
    pass

def c1c2(a):
    pass




