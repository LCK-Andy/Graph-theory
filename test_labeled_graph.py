from Graph import *

g = GraphList(["A", "B", "C", "D", "E", "F", "G"], directed=False)

g.add_edge("A", "B", 3)
g.add_edge("B", "C", 6)
g.add_edge("C", "D", 8)
g.add_edge("D", "E", 4)
g.add_edge("E", "F", 3)
g.add_edge("F", "G", 2)
g.add_edge("G", "A", 2)
g.add_edge("B", "G", 5)
g.add_edge("B", "F", 1)
g.add_edge("B", "E", 1)
g.add_edge("C", "E", 2)
g.add_edge("D", "G", 9)

# g.show("graph.html")
# g.optimal_spanning_tree(maximize=False).show("mst.html")
# print(g.optimal_spanning_tree(maximize=False).get_weight_sum())

print(g.get_degree_sequence())