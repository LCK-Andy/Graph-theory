from pyvis.network import Network

class GraphList:
    """
    create a graph using adjacency list
    """

    def __init__(self, n: int, directed=False):
        self.V_count = n
        self.E_count = 0
        self.directed = directed
        self.graph = [[] for _ in range(n)]

    def add_edge(self, From: int, To: int, weight=1):
        self.graph[From].append((To, weight))
        self.E_count += 1
        if not self.directed:
            self.graph[To].append((From, weight))

    def remove_edge(self, From: int, To: int, weight=None):
        if weight is None:
            # Simple graph case
            self.graph[From] = [edge for edge in self.graph[From] if edge[0] != To]
            self.E_count -= 1
            if not self.directed:
                self.graph[To] = [edge for edge in self.graph[To] if edge[0] != From]
        else:
            # Multi-graph case
            self.graph[From].remove((To, weight))
            self.E_count -= 1
            if not self.directed:
                self.graph[To].remove((From, weight))

    def print_graph(self):
        for i in range(self.V_count):
            print(i, ":", self.graph[i])

    def get_edge_list(self):
        edges = []
        for i in range(self.V_count):
            for j in self.graph[i]:
                if (j, i) in edges or (i, j) in edges and not self.directed:
                    continue
                edges.append((i, j[0], j[1]))
        return edges

    def show(self, filename="network.html"):
        net = Network(directed=self.directed)

        # add nodes
        for i in range(self.V_count):
            net.add_node(i, label=f"Node {i}")

        # add edges
        for i in self.get_edge_list():
            net.add_edge(i[0], i[1], value=i[2])

        net.show(filename, notebook=False)

    def get_node_degree(self, node: int):
        # return the degree of a node
        if not self.directed:
            return len(self.graph[node])
        else:
            return self.get_node_indegree(node) + self.get_node_outdegree(node)

    def get_node_indegree(self, node: int):
        # return the indegree of a node if the graph is directed
        if self.directed:
            indegree = 0
            for i in range(self.V_count):
                for edge in self.graph[i]:
                    if edge[0] == node:
                        indegree += 1
            return indegree
        else:
            return None

    def get_node_outdegree(self, node: int):
        # return the outdegree of a node if the graph is directed
        if self.directed:
            return len(self.graph[node])
        else:
            return None

    def get_degree_sequence(self):
        # return the degree sequence of the graph
        degree_sequence = []
        for i in range(self.V_count):
            degree_sequence.append(self.get_node_degree(i))

        return sorted(degree_sequence, reverse=True)

    def is_subgraph_of(self, graph):
        # use bfs to check if the graph is a subgraph of another graph
        # get a graph by deleting a random node from the graph
        # if the graph is isomorphic to the original graph, return True
        # else, continue the process until the graph is isomorphic or all nodes are deleted
        def bfs_util(queue, visited, graph):
            pass

        visited = set()

        pass

    def get_weight_sum(self):
        # return the sum of all edge weights in the graph
        weight_sum = 0
        for i in range(self.V_count):
            for edge in self.graph[i]:
                weight_sum += edge[1]
        if not self.directed:
            weight_sum /= 2

        return weight_sum

    def optimal_spanning_tree(self, maximize=False):
        # Kuruskal's algorithm
        # source: https://www.youtube.com/watch?v=_UH0H4r7N7E&t=25s
        result = GraphList(self.V_count, directed=self.directed)
        edges = self.get_edge_list()
        edges.sort(key=lambda x: x[2], reverse=maximize)

        # problem:
        # because it only checked if the nodes are visited or not,
        # this implementation may result in separate trees
        #
        # visited = set()
        # for edge in edges:
        #     if edge[0] not in visited or edge[1] not in visited:
        #         result.add_edge(From=edge[0], To=edge[1], weight=edge[2])
        #         visited.add(edge[0])
        #         visited.add(edge[1])
        #
        # suggestion:
        # use union-find algorithm to connect the separateed trees

        def find(parent, i):
            """
            the "find" function of the union-find algorithm

            source: https://www.youtube.com/watch?v=ayW5B2W9hfo
            """

            if parent[i] == i:
                return i
            else:
                return find(parent, parent[i])

        def union(parent, rank, x, y):

            """
            the "union" function of the union-find algorithm

            source: https://www.youtube.com/watch?v=ayW5B2W9hfo
            """
            root_x = find(parent, x)
            root_y = find(parent, y)

            # When two sets are merged, the root of the tree with a smaller
            # rank is made a child of the root of the tree with a larger rank.
            if rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            elif rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_y] = root_x
                rank[root_x] += 1

        parent = [] # used to keep track of the parent of each node
        rank = [] # used to keep track of the depth of each subtree rooted at each node

        for node in range(self.V_count):
            parent.append(node) # iitially, each node is its own parent
            rank.append(0) # initally , each subtree has a depth of 0

        for edge in edges:
            u, v, w = edge
            root_u = find(parent, u)
            root_v = find(parent, v)

            if root_u != root_v:
                result.add_edge(From=u, To=v, weight=w)
                union(parent, rank, root_u, root_v)
                
            # after the tree is found, root_u and root_v will be the same

        return result

    @staticmethod
    def is_graphical(sequence: list, mutiple_edges=False):
        sequence = sequence.copy()
        sequence.sort(reverse=True)

        # if there is a negative number in the sequence, it is not graphical
        if any([x < 0 for x in sequence]):
            return False

        # 1st test: the sum of the degrees must be even
        if mutiple_edges:
            if sum(sequence) % 2 == 0:
                return False
            else:
                return True

        # Havel-Hakimi algorithm
        while sequence:
            if sequence == [0]:
                return True

            # remove the first element
            n = sequence.pop(0)

            # if n is larger than the remaining sequence, it is not graphical
            if n >= len(sequence):
                return False

            # remove the next n elements
            for i in range(n):
                sequence[i] -= 1
                if sequence[i] < 0:
                    return False

            # sort the sequence in descending order
            sequence.sort(reverse=True)


def is_isomorphic(graph1: GraphList, graph2: GraphList):
    # test 1: same number of vertices and edges
    V1 = graph1.V_count
    V2 = graph2.V_count
    E1 = graph1.E_count
    E2 = graph2.E_count
    if V1 != V2 or E1 != E2:
        return False
    
    # test 2: same degree sequence
    degree_sequence1 = graph1.get_degree_sequence()
    degree_sequence2 = graph2.get_degree_sequence()
    if degree_sequence1 != degree_sequence2:
        return False
    
    # test 3: same subgraph structure
    # implement later
    
    return True


import numpy as np


class GraphMatrix:
    def __init__(self, n, directed=False):
        self.V_count = n
        self.E_count = 0
        self.directed = directed
        self.graph = np.zeros((n, n), dtype=int)

    def add_edge(self, From, To, weight=1):
        self.graph[From][To] = weight
        self.E_count += 1
        if not self.directed:
            self.graph[To][From] = weight

    def print_graph(self):
        print("     ", end="")
        for i in range(self.V_count):
            print(i, end=" ")
        print()
        for i in range(self.V_count):
            print(i, ":", self.graph[i])


class Path:
    """
    Path class for pathfinding based on a given GraphList object.
    """

    def __init__(self, graph_list: GraphList):
        self.V_count = graph_list.V_count
        self.directed = graph_list.directed
        self.graph = graph_list.graph

    def dfs(self, start, end, path=None):
        if path is None:
            path = []

        path = path + [start]

        if start == end:
            return path

        for node in self.graph[start]:
            if node not in path:
                new_path = self.dfs(node, end, path)
                if new_path:
                    return new_path

        return None

    def find_all_paths(self, start, end, path=None):
        if path is None:
            path = []

        path = path + [start]

        if start == end:
            return [path]

        paths = []
        for node in self.graph[start]:
            if node not in path:
                new_paths = self.find_all_paths(node, end, path)
                for p in new_paths:
                    paths.append(p)

        return paths


class CycleGraph(GraphList):
    """
    CycleGraph class to create a cycle graph using an adjacency list.

    In a cycle graph, each node is connected to exactly two other nodes,
    forming a single cycle. This class can create both directed and
    undirected cycles based on the provided direction.

    Attributes:
        n (int): Number of nodes in the graph.
        direction (str): Direction of the cycle. It can be "ascending"
                         for a directed cycle where edges go from lower
                         to higher numbered nodes, or "descending" for
                         a directed cycle where edges go from higher
                         to lower numbered nodes. Any other value will
                         create an undirected cycle.
    """
    def __init__(self, n, direction=None):
        if direction == "ascending" or direction == "descending":
            super().__init__(n, directed=True)
        else:
            super().__init__(n, directed=False)

        # Add edges to create a cycle
        if direction == "ascending":
            for i in range(n - 1):
                self.add_edge(i, i + 1)
            self.add_edge(n - 1, 0)
        else:
            for i in range(n - 1, 0, -1):
                self.add_edge(i, i - 1)
            self.add_edge(0, n - 1)


class CompleteGraph(GraphList):
    """
    CompleteGraph class to create a complete graph using an adjacency list.

    In a complete graph, every pair of distinct vertices is connected by a unique edge.

    Attributes:
        n (int): Number of nodes in the graph.
    """
    def __init__(self, n):
        super().__init__(n, directed=False)

        # add edges to create a complete graph
        for i in range(n):
            for j in range(i + 1, n):
                self.add_edge(i, j)

class StarGraph(GraphList):
    """
    StarGraph class to create a star-shaped graph using an adjacency list.
    
    In a star graph, one central node (node 0) is connected to all other nodes.
    
    Attributes:
        n (int): Number of nodes in the graph.
    """
    def __init__(self, n):
        super().__init__(n, directed=False)

        # add edges to create a star graph
        for i in range(1, n):
            self.add_edge(0, i)
