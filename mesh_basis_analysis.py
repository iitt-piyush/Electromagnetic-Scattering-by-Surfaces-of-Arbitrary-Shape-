import numpy as np
from collections import defaultdict
class Mesh_Basis_Analysis:
    def __init__(self, file_path, quadrature_rules, degree):
        self.file_path = file_path
        self.quadrature_rules = quadrature_rules
        self.degree = degree
        self.vertices = None
        self.elements = None
        self.internal_edges = None
        self.adjacent_cells = None
        self.edge_third_vertices = None
        self.vertex_to_triangles = None
        self.neighborhood = None
        self.mapped_gauss_points = None
        self.basis_function_values = None

    def read_custom_msh(self):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
        
        vertices_section = False
        elements_section = False
        vertices = []
        elements = []
        
        for line in lines:
            line = line.strip()
            
            if line == "Coordinates":
                vertices_section = True
                continue
            if line == "End Coordinates":
                vertices_section = False
                continue
            
            if line == "Elements":
                elements_section = True
                continue
            if line == "End Elements":
                elements_section = False
                continue
            
            if vertices_section:
                parts = line.split()
                if len(parts) == 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            
            if elements_section:
                parts = line.split()
                if len(parts) == 4:
                    elements.append([int(parts[1]), int(parts[2]), int(parts[3])])
        
        self.vertices = np.array(vertices)
        self.elements = np.array(elements)

    def find_internal_edges_and_third_vertices(self):
        vertices_indices = self.elements
        edge_triangle_map = {}
        vertex_to_triangles = defaultdict(set)
        
        for triangle_index, vertices in enumerate(vertices_indices):
            for i in range(3):
                vertex1 = vertices[i]
                vertex2 = vertices[(i + 1) % 3]
                edge = tuple(sorted([vertex1, vertex2]))
                if edge in edge_triangle_map:
                    edge_triangle_map[edge].append(triangle_index + 1)
                else:
                    edge_triangle_map[edge] = [triangle_index + 1]
                
                vertex_to_triangles[vertex1].add(triangle_index + 1)
        
        internal_edges = []
        adjacent_cells = []
        edge_third_vertices = []
        edge_index = 1

        for edge, triangles in edge_triangle_map.items():
            if len(triangles) == 2:
                adjacent_cells.append([edge_index] + triangles)
                t1, t2 = triangles
                t1_vertices = set(vertices_indices[t1 - 1])
                t2_vertices = set(vertices_indices[t2 - 1])
                third_vertex_t1 = list(t1_vertices - set(edge))[0]
                third_vertex_t2 = list(t2_vertices - set(edge))[0]
                edge_third_vertices.append((edge_index, third_vertex_t1, third_vertex_t2))
            if len(triangles) > 1:
                internal_edges.append((edge_index,) + edge)
                edge_index += 1

        self.internal_edges = internal_edges
        self.adjacent_cells = adjacent_cells
        self.edge_third_vertices = edge_third_vertices
        self.vertex_to_triangles = vertex_to_triangles

    def build_neighborhood(self):
        neighborhood = defaultdict(set)
        
        for triangle_index, vertices in enumerate(self.elements):
            triangle_id = triangle_index + 1
            for vertex in vertices:
                neighboring_triangles = self.vertex_to_triangles[vertex]
                for neighbor in neighboring_triangles:
                    if neighbor != triangle_id:
                        neighborhood[triangle_id].add(neighbor)

            neighborhood[triangle_id].add(triangle_id)
        
        self.neighborhood = {k: list(v) for k, v in neighborhood.items()}

    def map_to_3d_triangle(self, vertices, s, t):
        x1, y1, z1 = vertices[0]
        x2, y2, z2 = vertices[1]
        x3, y3, z3 = vertices[2]
        
        # mapping gauss points from unit isoceles triangle to 3_D oriented triangle 
        x_mapped = x1 + (x2 - x1) * s + (x3 - x1) * t
        y_mapped = y1 + (y2 - y1) * s + (y3 - y1) * t
        z_mapped = z1 + (z2 - z1) * s + (z3 - z1) * t
        
        mapped_point = np.array([x_mapped, y_mapped, z_mapped])
        
        return mapped_point

    def compute_mapped_gauss_points(self):
        mapped_gauss_points = {}
        
        for i, element in enumerate(self.elements):
            v1 = self.vertices[element[0] - 1]
            v2 = self.vertices[element[1] - 1]
            v3 = self.vertices[element[2] - 1]
            triangle_vertices = np.array([v1, v2, v3])
            
            gauss_points = []
            
            for s, t in self.quadrature_rules[self.degree]['points']:
                mapped_point = self.map_to_3d_triangle(triangle_vertices, s, t)
                gauss_points.append(mapped_point)
            
            mapped_gauss_points[i + 1] = np.array(gauss_points)
        
        self.mapped_gauss_points = mapped_gauss_points

    def calculate_basis_function(self):
        basis_function_values = {}

        for edge_info in self.adjacent_cells:
            edge_index = edge_info[0]
            triangle_plus = edge_info[1]
            triangle_minus = edge_info[2]

            gauss_points_plus = self.mapped_gauss_points[triangle_plus]
            gauss_points_minus = self.mapped_gauss_points[triangle_minus]

            third_vertex_plus = self.vertices[self.edge_third_vertices[edge_index - 1][1] - 1]
            third_vertex_minus = self.vertices[self.edge_third_vertices[edge_index - 1][2] - 1]

            _, v1_idx, v2_idx = self.internal_edges[edge_index - 1]
            v1 = self.vertices[v1_idx - 1]
            v2 = self.vertices[v2_idx - 1]
            ln = np.linalg.norm(v2 - v1)

            bf_values = {
                'length': ln,
                'p_plus': [],
                'p_minus': []
            }

            for g_point in gauss_points_plus:
                p_plus = g_point - third_vertex_plus
                bf_values['p_plus'].append(p_plus)

            for g_point in gauss_points_minus:
                p_minus = third_vertex_minus - g_point
                bf_values['p_minus'].append(p_minus)

            bf_values['p_plus'] = np.array(bf_values['p_plus'])
            bf_values['p_minus'] = np.array(bf_values['p_minus'])

            basis_function_values[edge_index] = bf_values

        self.basis_function_values = basis_function_values

    def run_analysis(self):
        self.read_custom_msh()
        self.find_internal_edges_and_third_vertices()
        self.build_neighborhood()
        self.compute_mapped_gauss_points()
        self.calculate_basis_function()

    def display_info(self):
        print(f"Vertices: {self.vertices}")
        print(f"Elements: {self.elements}")
        print(f"Internal Edges: {self.internal_edges}")
        print(f"Adjacent Cells: {self.adjacent_cells}")
        print(f"Edge Third Vertices: {self.edge_third_vertices}")
        print(f"Neighborhood: {self.neighborhood}")
        print(f"Mapped Gauss Points: {self.mapped_gauss_points}")
        print(f"Basis Function Values: {self.basis_function_values}")

