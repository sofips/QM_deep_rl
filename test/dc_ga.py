import numpy as np
#from dgamod import *
from zhang_actions_mod import *
import csv
import pygad
import sys
import time
import os
import configparser


# get parameters from config file
thisfolder = os.path.dirname(os.path.abspath(__file__))
initfile = os.path.join(thisfolder, str(sys.argv[1]))
print(initfile)
config = configparser.ConfigParser()
config.read(initfile)


# system parameters
n = config.getint('system_parameters', 'n')
dt = config.getfloat('system_parameters', 'dt')
b = config.getfloat('system_parameters', 'b')

acciones = acciones(b,n)
props = gen_props(acciones,n,b,dt)

fidelity_args = [props]


# genetic algorithm parameters
num_generations = config.getint('ga_initialization', 'num_generations')
num_genes = config.getint('ga_initialization', 'num_genes')
sol_per_pop = config.getint('ga_initialization', 'sol_per_pop')
fidelity_tolerance = config.getfloat('ga_initialization', 'fidelity_tolerance')
saturation = config.getint('ga_initialization', 'saturation')

# crossover and parent selection

num_parents_mating = config.getint('parent_selection', 'num_parents_mating')
parent_selection_type = config.get('parent_selection', 'parent_selection_type')
keep_elitism = config.getint('parent_selection', 'keep_elitism')
crossover_type = config.get('crossover', 'crossover_type')
crossover_probability = config.getfloat('crossover', 'crossover_probability')

# mutation
mutation_probability = config.getfloat('mutation', 'mutation_probability')
mutation_num_genes = config.getint('mutation', 'mutation_num_genes')

# call construction functions
on_generation = generation_func
fitness_func = fitness_func_constructor(fidelity, fidelity_args)
mutation_type = 'scramble'

gene_space = np.arange(0,16,1)
gene_type  = int

stop_criteria = ['saturate_'+str(saturation), 'reach_'+str(fidelity_tolerance)]

dirname = config.get('saving', 'directory')
n_samples = config.getint('saving', 'n_samples')

filename = dirname + '/nvsmaxfid.dat'



with open(filename, 'a') as f:
    for i in range(n_samples):
        writer = csv.writer(f,  delimiter=' ')
        
        #solutions_fname = dirname + '/act_sequence_sample'+ str(i)+'.dat'
        #fitness_history_fname = dirname + '/fitness_history_sample'+ str(i) + '.dat'
        
        t1 = time.time()

        initial_instance = pygad.GA(num_generations=num_generations,
                                    num_parents_mating=num_parents_mating,
                                    fitness_func=fitness_func,
                                    sol_per_pop=sol_per_pop,
                                    num_genes=num_genes,
                                    parent_selection_type=parent_selection_type,
                                    keep_elitism=keep_elitism,
                                    gene_space=gene_space,
                                    gene_type=gene_type,
                                    crossover_type=crossover_type,
                                    crossover_probability=crossover_probability,
                                    mutation_type=mutation_type,
                                    on_generation=on_generation,
                                    mutation_num_genes=mutation_num_genes,
                                    mutation_probability=mutation_probability,
                                    stop_criteria=stop_criteria,
                                    )

        initial_instance.run()

        t2 = time.time()
        trun = t2-t1

        maxg = initial_instance.generations_completed

        solution, solution_fitness, solution_idx = initial_instance.best_solution()

        row = [i, format(
            fidelity(solution,props)), '{:.8f}'.format(trun), maxg]
        writer.writerow(row)

        row = solution
        writer.writerow(row)

        #couplings_to_file(solution, solutions_fname, 'w')
        #fitness_history_to_file(initial_instance, fitness_history_fname)