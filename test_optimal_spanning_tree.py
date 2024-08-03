from Graph import *

g = GraphList(7)

g.add_edge(0, 1, 3)
g.add_edge(1, 2, 6)
g.add_edge(2, 3, 8)
g.add_edge(3, 4, 4)
g.add_edge(4, 5, 3)
g.add_edge(5, 6, 2)
g.add_edge(6, 0, 2)
g.add_edge(1, 6, 5)
g.add_edge(1, 5, 1)
g.add_edge(1, 4, 1)
g.add_edge(2, 4, 2)
g.add_edge(3, 6, 9)


g.show("graph.html")
g.optimal_spanning_tree(maximize=False).show("mst.html")
g.optimal_spanning_tree(maximize=False).print_graph()
print(g.optimal_spanning_tree(maximize=False).get_weight_sum())
