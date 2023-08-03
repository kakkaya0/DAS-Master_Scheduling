import Data_Generator as tsp
from Ham_Path import S_TSP as stsp
import networkx as nx
import gurobipy as gp
import ast
import Probabilities as prob
import random
import numpy as np
import matplotlib.pyplot as plt


# This method returns the node order for the whole solution route
def find_node_order_whole_route(instance):
    #Build the adjacency dictionary, this dictionary shows us the possible 
    # nodes we can travel to from each node, example: from (6,6) we can go to (5,5),(11,11) etc.

    #TO-DO: If I have n nodes, I should have n-1 edges, assert that, check that

    edges = tsp.load_route_info(instance)[3]

    adjacency_dict = {}
    for edge in edges:
        node1, node2 = edge
        if node1 not in adjacency_dict:
            adjacency_dict[node1] = set()
        if node2 not in adjacency_dict:
            adjacency_dict[node2] = set()
        adjacency_dict[node1].add(node2)
        adjacency_dict[node2].add(node1)

    # Initialize the list to store the node order
    node_order = []

    # Set to keep track of visited nodes
    visited = set()

    # Add nodes to the node order list
    # This method checks all the neighbors(the nodes you can visit from the chosen node)
    # and track whether we have already visited that node or not if that node is already visited
    # our route if there is an edge between the last node of the route and the node to add
    # otherwise add it to the beginning
    def add_to_node_order(node):
     visited.add(node)
     if len(node_order) > 0:
        last_node = node_order[-1]
        if last_node in adjacency_dict and node not in adjacency_dict[last_node]:
            node_order.insert(0,node)
            return
     node_order.append(node)
     if node in adjacency_dict:
        for neighbor in adjacency_dict[node]:
            if neighbor not in visited:
                add_to_node_order(neighbor)

    # This is to trigger the add_to_node_order method in the beginning
    # And in the end to check if all of the nodes in the given route are visited or not
    for edge in edges:
        node1, node2 = edge
        if node1 not in visited:
            add_to_node_order(node1)
        if node2 not in visited:
            add_to_node_order(node2)

    return node_order

#Example usage
#print(find_node_order_whole_route(24))




#This method returns the segments for the solution route
def get_segments(instance) :

#Loading the node order of the solution route and the compulsory stops
    route_order = find_node_order_whole_route(instance)
    compulsory_stops = tsp.load_route_info(instance)[0]

# Initialize the list to sort the compulsory stops based on their occurence in
# the route order list
    sorted_compulsory_stops = list()

# Sort the compulsory stops
    for node in route_order:
        for stop in compulsory_stops:
            if node == stop:
                sorted_compulsory_stops.append(node)

# Initialize a list of lists to find the segments
    segments  = []
    start_index = 0

# This for loop slices the route_order list based on the compulsory stops
# Each segment starts with a compulsory stop
# In the end if there are any additional nodes left,(nodes in the route order list before the first compulsory stop)
# We add those nodes to the last segment
    for i in range(0,len(sorted_compulsory_stops)):
        start_index = route_order.index(sorted_compulsory_stops[i])
        if i != len(sorted_compulsory_stops) - 1:
         end_index = route_order.index(sorted_compulsory_stops[i+1])
         segments.append(route_order[start_index:end_index])
         start_index = end_index
        else:
            end_index = len(route_order) - 1
            segments.append(route_order[start_index:end_index])

    for node in route_order[0:route_order.index(sorted_compulsory_stops[0])]:
        segments[len(segments)-1].append(node)


    return segments                





#Example Usage

#print(get_segments(24)[2])
#print(get_segments(24)[3])


#graph = tsp.generate_graph(24)[0]

#subgraph = graph.subgraph([(0,0),(1,1),(2,2)])

#print(subgraph.nodes)
#print(subgraph.edges)



# T0-D0 : A method to decide the shortest hamiltonian path and a method based on the solution

# This method returns the shortest hamiltonian path between two segments, order of nodes and the travel time
def shortest_hamiltonian_path_between_two_segments(instance, segment_index, speed):
    graph = tsp.generate_graph(instance)[0] #Load the complete graph based on the instance
    segments = get_segments(instance) # Get the segments of the solution route
    origin_segment = segments[segment_index] #Initialize the origin segment
    destination_segment = segments[segment_index + 1] # Initialize the destination segment
    compulsory_stops = list()
    optional_stops = list()
    
    # Create the route to calculate by adding the first element of the destination segment (the destination)
    # to the origin segment
    origin_segment.append(destination_segment[0])
    route_with_all_stops = origin_segment

# Initializing the origin and the destination
    origin = route_with_all_stops[0]
    destination = route_with_all_stops[len(route_with_all_stops) - 1]

# Differentiate the compulsory stops and optional stops
    compulsory_stops.append(origin)
    compulsory_stops.append(destination)

    optional_stops = [x for x in route_with_all_stops if x != origin and x != destination]
    active_optional_stops = list()

# Randomize the results to see whether an optional stop is active or not
# If active we are appending it to the active optional stops list
    for stop in optional_stops:

        probabilities = [prob.get_prob_being_non_active(instance,stop), prob.get_prob_being_active(instance,stop)]
        result = random.choices([0,1],probabilities)[0]

        if result == 1:
            active_optional_stops.append(stop)
    
    route_with_active_stops = list()
    route_with_active_stops.append(origin)
    route_with_active_stops.extend(active_optional_stops)
    route_with_active_stops.append(destination)


# Get the subgraph based on the nodes in the newly created route_to_calculate
# This gives us a subgraph only containing the relevant nodes and edges
# This is needed to get the distance matrix
    subgraph = graph.subgraph(route_with_active_stops)

# Create another subgraph with the edges and nodes from the solution route
# Again this gives us a subgraph only containing the relevant edges and nodes from the solution route
    subgraph_route = nx.DiGraph()
    subgraph_route.add_edges_from(tsp.load_route_info(instance)[3])
    subgraph_route = subgraph.subgraph(route_with_active_stops)

# Create the MIP-model and optimize it
    self = stsp(subgraph,subgraph_route.edges,subgraph_route.nodes,origin,destination)
    self.calculate_dist_dict()
    self.build_model()
    self.solve_model()

#Get the selected edges as a list
    if self.m.status == gp.GRB.OPTIMAL:
     selected_edges = []
    # Iterate over the variables
     for variable in self.m.getVars():
        # Check if the variable is selected (has a positive value)
        if variable.x > 0.5:
            # Extract the edge from the variable name and add it to the list
            edge = variable.varName.split('_')[1]  # Assuming the variable name follows a specific format
            selected_edges.append(edge)

#  Convert the elements of selected edges list to tuples from strings  
     selected_edges = [ast.literal_eval(string) for string in selected_edges]

# This method is explained inside the find_node_order_whole_route method
     adjacency_dict = {}
    for edge in selected_edges:
        node1, node2 = edge
        if node1 not in adjacency_dict:
            adjacency_dict[node1] = set()
        if node2 not in adjacency_dict:
            adjacency_dict[node2] = set()
        adjacency_dict[node1].add(node2)
        adjacency_dict[node2].add(node1)

    # Initialize the list to store the node order
    node_order = []

    # Set to keep track of visited nodes
    visited = set()

    # Add nodes to the node order list
    # This method checks all the neighbors(the nodes you can visit from the chosen node)
    # and track whether we have already visited that node or not if that node is already visited
    # we check the next neighbor otherwise we add it to our node order list as the next element of
    # our route if there is an edge between the last node of the route and the node to add
    # otherwise add it to the beginning
    def add_to_node_order(node):
     visited.add(node)
     if len(node_order) > 0:
        last_node = node_order[-1]
        if last_node in adjacency_dict and node not in adjacency_dict[last_node]:
            node_order.insert(0,node)
            return
     node_order.append(node)
     if node in adjacency_dict:
        for neighbor in adjacency_dict[node]:
            if neighbor not in visited:
                add_to_node_order(neighbor)

    # This is to trigger the add_to_node_order method in the beginning
    # And in the end to check if all of the nodes in the given route are visited or not
    for edge in selected_edges:
        node1, node2 = edge
        if node1 not in visited:
            add_to_node_order(node1)
        if node2 not in visited:
            add_to_node_order(node2)

 # Calculate the total weight of the shortest hamiltonian path   
    total_weight = 0

    for edge in selected_edges:
        total_weight = total_weight + self.dist.get(edge)

 # Calculate the travel time based on the speed parameter  
    travel_time = total_weight/speed

    


    

 
    return selected_edges,node_order,total_weight,travel_time



#Example usage
#subgraph = shortest_hamiltonian_path_between_two_segments(24,1,35)
#print(subgraph)



#TO-DO : take samples for each sample calculate pmf,cdf and arrival time
# If standard deviation is close to mean then stop


def single_segment_subproblem(instance,segment_index,speed,number_of_samples,dimension_of_samples,standard_deviation,epsilon):

#For single segment subproblem we assume that lower bound (Ah) is 0
 lower_bound = 0

#Create sampling results dictionary for Total Weight and Travel Time in each dimension
 sampling_results = {}
 value_names = ['Total Weight', 'Travel Time']

#Create a list to save each dimension in that.
 all_samples_list = list()


#Iterate through each dimension of every sample and calculate the total weight and travel time
#They are different for each dimension because optional stops are randomized depending on the probability distribution of demand
 for j in range(number_of_samples):
  sampling_results = {}
  value_names = ['Total Weight', 'Travel Time']
  for i in range(dimension_of_samples):

     method_results = shortest_hamiltonian_path_between_two_segments(instance,segment_index,speed)
     total_weight = method_results[2]
     travel_time = method_results[3]

     sampling_results[i] = {
        value_names[0]: total_weight,
        value_names[1]: travel_time 
        }
  all_samples_list.append(sampling_results)

# Create a new dictionary to calculate the service times and upper bounds for each sample
 service_times = {}
 value_names_service = ['Service Time(Hh)', 'Upper Bound(Bh)']

 #For-loop to iterate through each sample   
 for i in range(number_of_samples):
    # Get the travel time values for each sample and build an histogram to calculate the Probability Mass Function(PMF) and Cumulative Distribution Function (CDF)
      travel_time_values = [all_samples_list[i][result][value_names[1]] for result in all_samples_list[i]]
      count, bins_count = np.histogram(travel_time_values, bins = 10)
    # Calculate pmf and cdf and then find the lowest service time value, which guarantees with a probability of 1-epsilon, that the vehicle has a sufficent time to serve the active set  
      pmf = count / sum(count)
      cdf = np.cumsum(pmf)
      given_probability = 1-epsilon
      service_time = np.interp(given_probability, cdf, bins_count[1:])

    #Upper bound is equal to lower bound + service time  
      upper_bound = lower_bound + service_time
    
    #Save the service time and upper bound for each sample in a dictionary
      service_times[i] = {
        value_names_service[0]: service_time,
        value_names_service[1]: upper_bound 
        }

#Create a list for upper bound values of each sample
#Iterate through every sample and get the upper_bound values
 upper_bound_values = list()
 for i in range(len(service_times)):
    upper_bound_values = [service_times[result][value_names_service[1]] for result in service_times]


#Calculate the mean and standard deviation of the upper bound value of every sample
 mean = np.mean(upper_bound_values)
 sd = np.std(upper_bound_values)

#If calculated standard deviation value is smaller than our standard deviation input (if it is close to the mean)
#Then the solution is precise and we return the mean and dimension of samples
# If it is bigger than standard deviation input then we run the method once again with a bigger cardinality of dimension size
 if (sd < standard_deviation) :

    return mean, dimension_of_samples

 else :
   return single_segment_subproblem(instance,segment_index,speed,number_of_samples, dimension_of_samples = dimension_of_samples + 10, standard_deviation = standard_deviation, epsilon = epsilon  )





#print(single_segment_subproblem(24,0,35,10,10,1,0.05))





