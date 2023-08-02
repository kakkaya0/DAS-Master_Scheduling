import Data_Generator as tsp
import networkx as nx

G = tsp.generate_graph(24)[0]
names=tsp.generate_graph(24)[1]

route_edges = tsp.load_route_info(24)[3]
compulsory_stops = tsp.load_route_info(24)[0]



# Extract the unique nodes from the edges
nodes = set()
for edge in route_edges:
    nodes.update(edge)

# Sort the nodes in the order they appear in the route
node_order = []
for edge in route_edges:
    for node in edge:
        if node in nodes:
            node_order.append(node)
            nodes.remove(node)

# Print the node order
print(node_order)

print(compulsory_stops)






