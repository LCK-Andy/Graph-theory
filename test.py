from Graph import *

g = GraphList(6)

g.add_edge(0, 1, 4)
g.add_edge(1, 2, 7)
g.add_edge(2, 3, 8)
g.add_edge(3, 4, 4)
g.add_edge(4, 5, 1)
g.add_edge(5, 0, 1)
g.add_edge(1, 5, 5)
g.add_edge(1, 3, 2)
g.add_edge(1, 4, 3)
g.add_edge(0, 4, 1)
g.add_edge(3, 5, 3)

print(g.print_graph())
