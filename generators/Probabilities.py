#calculate probabilities
#probability of request being 0 in each node (node can be origin or destination)
#and then 1-that probability


import Data_Generator as tsp
import numpy
import random
import matplotlib.pyplot as plt 
import networkx as nx
import itertools
import gurobipy as gp
from gurobipy import GRB


#calculating the probability of a node being active from the given probabilities

def get_prob_being_active(instance, node):
    #create a new list to save distance matrix and demand dict under it
    list_demand_dist = list()

    graph_names = tsp.generate_graph(instance)[1]

#load the instance
    list_demand_dist = tsp.load_demand(instance)

#divide the list as distance matrix and demand dict

    dist_matrix = list_demand_dist[0]
    demand_dict = list_demand_dist[1]

#create an empty list to use after
    list_v1 = list()

#get all the keys from demand dict
    demand_dict_keys = demand_dict.keys()

    for key, value in graph_names.items():
     if value == node:
       node_key = key

#iterate through all the keys and get all the values, 
# which has the node has a origin or destination
    for key in demand_dict_keys:
        if node_key in key:
            list_v1.append(demand_dict[key])

#calculate the probability of that node being non-active
    p_non_active = 1

    for item in list_v1:
        p_non_active = p_non_active * item['probability'][0]

   #1 - prob of being non_active equals to probability of that node being active 
    p_active = 1-p_non_active
    return p_active

print(get_prob_being_active(24,(1,1)))
    
#calculating the probability of a node being non_active from the given probabilities

def get_prob_being_non_active(instance,node):

        #create a new list to save distance matrix and demand dict under it
    list_demand_dist = list()

#load the instance
    list_demand_dist = tsp.load_demand(instance)

#divide the list as distance matrix and demand dict

    dist_matrix = list_demand_dist[0]
    demand_dict = list_demand_dist[1]

#create an empty list to use after
    list_v1 = list()

#get all the keys from demand dict
    demand_dict_keys = demand_dict.keys()

#iterate through all the keys and get all the values, 
# which has the node has a origin or destination
    for key in demand_dict_keys:
        if node in key:
            list_v1.append(demand_dict[key])

#calculate the probability of that node being non-active
    p_non_active = 1

    for item in list_v1:
        p_non_active = p_non_active * item['probability'][0]

    return p_non_active




#calculate a probability of a set being active,

