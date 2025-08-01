from mpcc_models.bard1 import Bard1Penalized

def test_penalized_bard1():
    model = Bard1Penalized(inner_penalty_init=1.0, outer_penalty_init=1.0)
    x0 = model.initial_guess()
    res, x, fn = model.solve(x0, print_level=0, verbose=True)
    print("Penalized Bard1 Solution:", x)
    print("Function value at solution:", fn)
    print("Optimization result:", res)
    