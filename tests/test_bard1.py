import numpy as np
from mpcc_models.bard1 import Bard1

def test_bard1():
    mpcc = Bard1(relaxation_type="scholtes")
    x0 = 0.1*np.ones(5)
    res = mpcc.solve(x0,verbose=True, print_level=1,max_iter=30)
    print(res)
