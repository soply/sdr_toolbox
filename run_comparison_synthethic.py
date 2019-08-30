# coding: utf8
"""
Run file to compare different SDR methods for a given synthethic problem.
Multiprocessing possible.
"""
# coding: utf8
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
# I/O import
import os
import shutil
import sys
import tempfile
import time

# Specific imports
import numpy as np
# For parallelization
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid

from problem_factory.problems import get_problem
from problem_factory.sampling import *

from sdr_handler import estimate_sdr

def run_example(N,
                D,
                sigma_f,
                random_seeds,
                problem_id,
                estimator_id,
                options,
                index_space_error, # File tangent error
                comp_time, # File computational time
                rep, i1, i2, i3):
    """
    Main function to run a single experiment. Saves the results into the
    given files.
    """
    np.random.seed(random_seeds[i1, i2, i3, rep])
    f, basis = get_problem(problem_id, D)
    d = basis.shape[1]
    X, Y = sample_data_uniform_ball(N, D, f, sigma_f)
    levelsets_to_test = options['params']['n_levelsets']
    if estimator_id in ['IHT', 'PHD']:
        # These estimators do not use the number of level sets as a parameter
        start = time.time()
        vecs = estimate_sdr(X, Y, method = estimator_id, d = d, **options)
        end = time.time()
        index_space_error[i1, i2, i3, :, rep] = np.linalg.norm(vecs.dot(vecs.T) - basis.dot(basis.T))
        comp_time[i1, i2, i3, :, rep] = end - start
    elif estimator_id in ['RCLR_proxy']:
        start = time.time()
        vecs, proxy = estimate_sdr(X, Y, method = estimator_id, d = d, max_n_levelsets = np.max(levelsets_to_test), **options)
        end = time.time()
        index_space_error[i1, i2, i3, :, rep] = np.linalg.norm(vecs.dot(vecs.T) - basis.dot(basis.T))
        comp_time[i1, i2, i3, :, rep] = end - start
    else:
        for i, j in enumerate(levelsets_to_test):
            start = time.time()
            vecs = estimate_sdr(X, Y, method = estimator_id, d = d, n_levelsets = j, **options)
            end = time.time()
            index_space_error[i1, i2, i3, i, rep] = np.linalg.norm(vecs.dot(vecs.T) - basis.dot(basis.T))
            comp_time[i1, i2, i3, i, rep] = end - start
    # Check if crossvalidation is done
    # Setting the test manifold, check synthethic_problem_factory.curves
    print "Finished N = {0}     D = {1}     sigma_f = {2}   rep = {3}".format(
        N, D, sigma_f, rep)


if __name__ == "__main__":
    # Get number of jobs from sys.argv
    if len(sys.argv) > 1:
        n_jobs = int(sys.argv[1])
    else:
        n_jobs = 1 # Default 1 jobs
    print 'Using n_jobs = {0}'.format(n_jobs)
    # Define manifolds to test
    problems_ids = ['simple_division', 'pw_lin2','exp', 'exp3', 'sincos', 'sincosdivided'] # Problems to test
    estimator_ids = ['RCLR_proxy']
    # Parameters
    run_for = {
        'N' : [50000],
        'D' : [20],
        'sigma_f' : [0.01], # Standard deviation of function error
        'repititions' : 3,
        # Estimator information
        'options' : {
            'split_by' : 'dyadic', # Inverse regression based techniques
            'use_residuals' : False, # IHT and PHD
            'whiten' : True,
            'return_mat' : False,
            'params' : {
                'n_levelsets' : [i + 1 for i in range(100)],
            }
        }
    }
    random_seeds = np.random.randint(0, high = 2**32 - 1, size = (len(run_for['N']),
                                                          len(run_for['D']),
                                                          len(run_for['sigma_f']),
                                                          run_for['repititions']))
    for problem_id in problems_ids:
        for estimator_id in estimator_ids:
            print "Considering problem {0} with estimator {1}".format(problem_id, estimator_id)
            savestr_base = 'rclr_proxy_test/'
            filename_errors = 'results/' + savestr_base + problem_id + '/' + estimator_id
            try:
                index_space_error = np.load(filename_errors + '/index_space_error.npy')
                comp_time = np.load(filename_errors + '/comp_time.npy')
            except IOError:
                if not os.path.exists(filename_errors):
                    os.makedirs(filename_errors)
                    # Save a log file
                with open(filename_errors + '/log.txt', 'w') as file:
                    file.write(json.dumps(run_for, indent=4)) # use `json.loads` to do the reverse
                tmp_folder = tempfile.mkdtemp()
                dummy_for_shape = np.zeros((len(run_for['N']),
                                            len(run_for['D']),
                                            len(run_for['sigma_f']),
                                            len(run_for['options']['params']['n_levelsets']),
                                            run_for['repititions']))
                try:
                    # Create error containers
                    index_space_error = np.memmap(os.path.join(tmp_folder, 'index_space_error'), dtype='float64',
                                               shape=dummy_for_shape.shape, mode='w+')
                    comp_time = np.memmap(os.path.join(tmp_folder, 'comp_time'), dtype='float64',
                                               shape=dummy_for_shape.shape, mode='w+')
                    # Run experiments in parallel
                    Parallel(n_jobs=n_jobs, backend = "multiprocessing")(delayed(run_example)(
                                        run_for['N'][i1],
                                        run_for['D'][i2],
                                        run_for['sigma_f'][i3],
                                        random_seeds,
                                        problem_id,
                                        estimator_id,
                                        run_for['options'],
                                        index_space_error,
                                        comp_time,
                                        rep, i1, i2, i3)
                                        for rep in range(run_for['repititions'])
                                        for i1 in range(len(run_for['N']))
                                        for i2 in range(len(run_for['D']))
                                        for i3 in range(len(run_for['sigma_f'])))
                    # Dump memmaps to files
                    index_space_error.dump(filename_errors + '/index_space_error.npy')
                    comp_time.dump(filename_errors + '/comp_time.npy')
                finally:
                    try:
                        shutil.rmtree(tmp_folder)
                    except:
                        print('Failed to delete: ' + tmp_folder)
