# coding: utf8
import numpy as np

def get_problem(identifier, D):
    if identifier == 'linreg':
        def f(x):
            return x[0]
        basis = np.zeros((D, 1))
        basis[0,0] = 1.0
    elif identifier == 'SIM':
        def f(x):
            return 0.25 * (x[0] + x[1]) + np.sin((x[0] + x[1])) + (x[0] + x[1]) ** 2
        basis = np.zeros((D, 1))
        basis[0,0] = 1.0/np.sqrt(2)
        basis[1,0] = 1.0/np.sqrt(2)
    elif identifier == 'SIM2':
        def f(x):
            return 1.0/(1.0 + np.exp(-x[0]))
        basis = np.zeros((D, 1))
        basis[0,0] = 1.0
    elif identifier == 'non_monoton_SIM':
        def f(x):
            return (x[0]-0.25) ** 2
        basis = np.zeros((D, 1))
        basis[0,0] = 1.0
    elif identifier == 'simple_division':
        def f(x):
            return x[1]/(1 + (x[0] - 1) ** 2)
        basis = np.zeros((D, 2))
        basis[0,0] = 1.0
        basis[1,1] = 1.0
    elif identifier == "sincos":
        def f(x):
            return np.sin(x[0]) + np.cos((x[1]-0.25))
        basis = np.zeros((D, 2))
        basis[0,0] = 1.0
        basis[1,1] = 1.0
    elif identifier == "sincosdivided":
        def f(x):
            return (np.sin(x[0]) + np.cos((x[1]-0.25)))/(1.0 + x[0] ** 2)
        basis = np.zeros((D, 2))
        basis[0,0] = 1.0
        basis[1,1] = 1.0
    elif identifier == "sincospol":
        def f(x):
            return np.sin(2 * np.pi * x[0]) + 2.05 * np.pi * x[0] + 6.0 * x[1]
        basis = np.zeros((D, 2))
        basis[0,0] = 1.0
        basis[1,1] = 1.0
    elif identifier == 'interaction':
        def f(x):
            return x[0] * x[1]
        basis = np.zeros((D, 2))
        basis[0,0] = 1.0
        basis[1,1] = 1.0
    elif identifier == "shifted_norm":
        def f(x):
            return (x[0]-0.25) ** 2 + (x[1] - 0.25) ** 2
        basis = np.zeros((D, 2))
        basis[0,0] = 1.0
        basis[1,1] = 1.0
    elif identifier == "norm_lin":
        def f(x):
            return (x[0]-0.25) ** 2 + 0.25 * x[1] + np.sin(x[2]-0.5)
        basis = np.zeros((D, 3))
        basis[0,0] = 1.0
        basis[1,1] = 1.0
        basis[2,2] = 1.0
    elif identifier == "sincoscos":
        def f(x):
            return np.sin(2 * np.pi * x[0]) * np.cos(2.0 * x[0] * x[1]) * np.cos(x[2])
        basis = np.zeros((D, 3))
        basis[0,0] = 1.0
        basis[1,1] = 1.0
        basis[2,2] = 1.0
    elif identifier == "pw_lin":
        def f(x):
            y = np.zeros(x.shape[1])
            def coord1(x):
                if x[0] < 0.0:
                    return 0.1 * x[0]
                else:
                    return 2.0 * (x[0] - 0.5) + 0.05
            def coord2(x):
                if x[1] < 0.0:
                    return 2.0 * x[1]
                else:
                    return 1.0 + 0.1 * (x[1] - 0.5)
            for i in range(x.shape[1]):
                y[i] = coord1(x[:,i]) + coord2(x[:,i])
            return y
        basis = np.zeros((D, 2))
        basis[0,0] = 1.0
        basis[1,1] = 1.0
    elif identifier == "pw_lin2":
        def f(x):
            y = np.zeros(x.shape[1])
            def coord1(x):
                if x[0] < 0.0:
                    return 0.1 * x[0]
                else:
                    return 2.0 * (x[0] - 0.5) + 0.05
            def coord2(x):
                if x[1] < 0.0:
                    return 2.0 * x[1]
                else:
                    return 1.0 + 0.1 * (x[1] - 0.5)
            def coord3(x):
                if x[2] < 0.0:
                    return 5.0 * x[2]
                else:
                    return 0.1 * (x[2] - 0.2) + 1.0
            for i in range(x.shape[1]):
                y[i] = coord1(x[:,i]) + coord2(x[:,i]) + coord3(x[:,i])
            return y
        basis = np.zeros((D, 3))
        basis[0,0] = 1.0
        basis[1,1] = 1.0
        basis[2,2] = 1.0
    elif identifier == "exp":
        def f(x):
            y = x[1] * np.exp(np.sin(x[0]) * x[1] + x[1])
            return y
        basis = np.zeros((D, 2))
        basis[0,0] = 1.0
        basis[1,1] = 1.0
    elif identifier == "exp3":
        def f(x):
            y = x[1] * np.exp(np.sin(x[0]) * x[1] + x[2])
            return y
        basis = np.zeros((D, 3))
        basis[0,0] = 1.0
        basis[1,1] = 1.0
        basis[2,2] = 1.0
    else:
        raise NotImplementedError("Problem {0} is not implemented".format(identifier))
    return f, basis
