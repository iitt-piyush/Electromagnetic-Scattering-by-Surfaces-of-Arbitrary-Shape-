import numpy as np
from scipy.sparse.linalg import cgs
import time 
from collections import defaultdict

def analytical_integral(point_m_minus, point_n_minus):
    def compute_triangle_properties(r,P1, P2, P3):
        # Convert points to numpy arrays
        P1 = np.array(P1)
        P2 = np.array(P2)
        P3 = np.array(P3)
        r  = np.array(r)
        
        # Calculate side lengths
        l1 = np.linalg.norm(P3 - P2)
        l2 = np.linalg.norm(P1 - P3)
        l3 = np.linalg.norm(P2 - P1)
        
        # Calculate normal vector and area
        n = np.cross(P2 - P1, P3 - P1)
        A = 0.5 * np.linalg.norm(n)
        n_hat = n / np.linalg.norm(n)
        
        # Calculate unit vectors
        u_hat = (P2 - P1) / l3
        v_hat = np.cross(n_hat, u_hat)
        w_hat = n_hat
        
        # Calculate the unit vector of a side
        s1_hat = (P3 - P2) / l1
        s2_hat = (P1 - P3) / l2
        s3_hat = (P2 - P1) / l3

        # Calculate the perpendicular unit vector
        m1_hat = np.cross(s1_hat, n_hat)
        m2_hat = np.cross(s2_hat, n_hat)
        m3_hat = np.cross(s3_hat, n_hat)

        # Calculate local coordinates of the third node
        u3 = np.dot(P3 - P1, u_hat)
        v3 = 2 * A / l3
        
        # Calculate local coordinates of observation point r
        u0 = np.dot(u_hat, r - P1)
        v0 = np.dot(v_hat, r - P1)
        w0 = np.dot(w_hat, r - P1) 
        
        #print("u0:",u0,"v0:",v0,"w0:",w0)
        
        # Return computed quantities
        return {
            'side_lengths': {'l1': l1, 'l2': l2, 'l3': l3},
            'normal_vector': n_hat,
            'area': A,
            'unit_vectors': {'u_hat': u_hat, 'v_hat': v_hat, 'w_hat': w_hat},
            'side_vectors': {'s1_hat': s1_hat, 's2_hat': s2_hat, 's3_hat': s3_hat, 'm1_hat': m1_hat, 'm2_hat': m2_hat, 'm3_hat': m3_hat},
            'local_coordinates_P3': {'u3': u3, 'v3': v3},
            'local_coordinates_r':  {'u0': u0, 'v0': v0, 'w0':w0}
        }

    def compute_si_values( triangle_props):

        # Extract triangle properties
        l1 = triangle_props['side_lengths']['l1']
        l2 = triangle_props['side_lengths']['l2']
        l3 = triangle_props['side_lengths']['l3']
        u3 = triangle_props['local_coordinates_P3']['u3']
        v3 = triangle_props['local_coordinates_P3']['v3']
        u0 = triangle_props['local_coordinates_r']['u0']
        v0 = triangle_props['local_coordinates_r']['v0']
        
        # Compute s values for each side
        
        # Side 1: P1 to P2
        s1_minus = -(((l3 - u0) * (l3 - u3) + v0 * v3) / l1)
        s1_plus = (((u3 - u0) * (u3 - l3) + v3 * (v3 - v0)) / l1)
        
        # Side 2: P2 to P3
        s2_minus = -((u3 * (u3 - u0) + v3 * (v3 - v0)) / l2)
        s2_plus = ((u0 * u3 + v0 * v3) / l2)
        
        # Side 3: P3 to P1
        s3_minus = -u0
        s3_plus = l3 - u0
        
        # Return s values
        return {
            's1_minus': s1_minus,
            's1_plus': s1_plus,
            's2_minus': s2_minus,
            's2_plus': s2_plus,
            's3_minus': s3_minus,
            's3_plus': s3_plus
        }
    def compute_distances_to_nodes( triangle_props):

        # Extract triangle properties
        l1 = triangle_props['side_lengths']['l1']
        l2 = triangle_props['side_lengths']['l2']
        l3 = triangle_props['side_lengths']['l3']
        u3 = triangle_props['local_coordinates_P3']['u3']
        v3 = triangle_props['local_coordinates_P3']['v3']
        u0 = triangle_props['local_coordinates_r']['u0']
        v0 = triangle_props['local_coordinates_r']['v0']

        # Calculate t1^0, t2^0, t3^0
        t1_0 = (v0 * (u3 - l3) + v3 * (l3 - u0)) / l1
        t2_0 = (u0 * v3 - v0 * u3) / l2
        t3_0 = v0

        # Compute distances to nodes
        t1_minus = np.sqrt((l3 - u0)**2 + v0**2)
        t1_plus = np.sqrt((u3 - u0)**2 + (v3 - v0)**2)
        
        t2_plus = np.sqrt(u0**2 + v0**2)
        
        t3_plus = t1_minus
        t2_minus = t1_plus
        t3_minus = t2_plus
        
        return {
            't1_minus': t1_minus,
            't1_plus': t1_plus,
            't2_plus': t2_plus,
            't3_plus': t3_plus,
            't2_minus': t2_minus,
            't3_minus': t3_minus,
            't1_0': t1_0,
            't2_0': t2_0,
            't3_0': t3_0
        }

    def compute_distances_to_node_from_r(triangle_props, si_values, dis_p):
        # Extract necessary triangle properties
        w0 = triangle_props['local_coordinates_r']['w0']
        
        # Extract distances to nodes and s values
        t1_0 = dis_p['t1_0']
        t2_0 = dis_p['t2_0']
        t3_0 = dis_p['t3_0']
        
        s1_minus = si_values['s1_minus']
        s1_plus = si_values['s1_plus']
        s2_minus = si_values['s2_minus']
        s2_plus = si_values['s2_plus']
        s3_minus = si_values['s3_minus']
        s3_plus = si_values['s3_plus']

        # Calculate distance R0 to the i-th side
        R1_0 = np.sqrt(t1_0**2 + w0**2)
        R2_0 = np.sqrt(t2_0**2 + w0**2)
        R3_0 = np.sqrt(t3_0**2 + w0**2)
    
        # Calculate distances Ri^- and Ri^+
        R1_minus = np.sqrt(R1_0**2 + s1_minus**2)
        R1_plus = np.sqrt(R1_0**2 + s1_plus**2)

        R2_minus = np.sqrt(R2_0**2 + s2_minus**2)
        R2_plus = np.sqrt(R2_0**2 + s2_plus**2)

        R3_minus = np.sqrt(R3_0**2 + s3_minus**2)
        R3_plus = np.sqrt(R3_0**2 + s3_plus**2)

        # Return distances   
        return {
            'R1_0': R1_0,
            'R2_0': R2_0,
            'R3_0': R3_0,
            'R1_minus': R1_minus,
            'R1_plus': R1_plus,
            'R2_minus': R2_minus,
            'R2_plus': R2_plus,
            'R3_minus': R3_minus,
            'R3_plus': R3_plus
        }

    def compute_f2i(Ri_plus, Ri_minus, si_plus, si_minus):
        f2i = np.log((Ri_plus + si_plus) / (Ri_minus + si_minus))
        return np.nan_to_num(f2i, nan=0.0, posinf=0.0, neginf=0.0)

    def compute_f3i(si_plus, si_minus, Ri_plus, Ri_minus, Ri_0, f2i):
        f3i = (si_plus * Ri_plus - si_minus * Ri_minus) + (Ri_0**2 * f2i)
        return np.nan_to_num(f3i, nan=0.0, posinf=0.0, neginf=0.0)

    def compute_bi(ti_0, si_plus, si_minus, Ri_plus, Ri_minus, Ri_0, w0):
        term1 = np.arctan(ti_0 * si_plus / (Ri_0**2 + np.abs(w0) * Ri_plus))
        term2 = np.arctan(ti_0 * si_minus / (Ri_0**2 + np.abs(w0) * Ri_minus))
        bi = term1 - term2
        return np.nan_to_num(bi, nan=0.0, posinf=0.0, neginf=0.0)

    def compute_function_set(triangle_props, si_values, dis_r, dis_p):
        # Extract necessary values
        w0 = triangle_props['local_coordinates_r']['w0']
        
        # Compute f2i for each side
        f2_1 = compute_f2i(dis_r['R1_plus'], dis_r['R1_minus'], si_values['s1_plus'], si_values['s1_minus'])
        f2_2 = compute_f2i(dis_r['R2_plus'], dis_r['R2_minus'], si_values['s2_plus'], si_values['s2_minus'])
        f2_3 = compute_f2i(dis_r['R3_plus'], dis_r['R3_minus'], si_values['s3_plus'], si_values['s3_minus'])
        
        # Compute f3i for each side
        f3_1 = compute_f3i(si_values['s1_plus'], si_values['s1_minus'], dis_r['R1_plus'], dis_r['R1_minus'], dis_r['R1_0'], f2_1)
        f3_2 = compute_f3i(si_values['s2_plus'], si_values['s2_minus'], dis_r['R2_plus'], dis_r['R2_minus'], dis_r['R2_0'], f2_2)
        f3_3 = compute_f3i(si_values['s3_plus'], si_values['s3_minus'], dis_r['R3_plus'], dis_r['R3_minus'], dis_r['R3_0'], f2_3)
        
        # Compute bi for each side compute_bi(ti0, si_plus, si_minus, Ri_plus, Ri_minus, R0, w0):
        b1 = compute_bi(dis_p['t1_0'], si_values['s1_plus'], si_values['s1_minus'], dis_r['R1_plus'], dis_r['R1_minus'], dis_r['R1_0'], w0)
        b2 = compute_bi(dis_p['t2_0'], si_values['s2_plus'], si_values['s2_minus'], dis_r['R2_plus'], dis_r['R2_minus'], dis_r['R2_0'], w0)
        b3 = compute_bi(dis_p['t3_0'], si_values['s3_plus'], si_values['s3_minus'], dis_r['R3_plus'], dis_r['R3_minus'], dis_r['R3_0'], w0)
        
        # Compute b
        b = b1 + b2 + b3

        return {
            'f2_1': f2_1,
            'f2_2': f2_2,
            'f2_3': f2_3,
            'f3_1': f3_1,
            'f3_2': f3_2,
            'f3_3': f3_3,
            'b1': b1,
            'b2': b2,
            'b3': b3,
            'b': b
        }

    def compute_I(triangle_props, function_set, dis_p, P1, P2, P3):
        # Extract necessary values
        w0 = triangle_props['local_coordinates_r']['w0']
        u_hat = triangle_props['unit_vectors']['u_hat']
        v_hat = triangle_props['unit_vectors']['v_hat']
        w_hat = triangle_props['unit_vectors']['w_hat']
        m1_hat = triangle_props['side_vectors']['m1_hat']
        m2_hat = triangle_props['side_vectors']['m2_hat']
        m3_hat = triangle_props['side_vectors']['m3_hat']
        u0 = triangle_props['local_coordinates_r']['u0']
        v0 = triangle_props['local_coordinates_r']['v0']
        f2_1 = function_set['f2_1']
        f2_2 = function_set['f2_2']
        f2_3 = function_set['f2_3']
        f3_1 = function_set['f3_1']
        f3_2 = function_set['f3_2']
        f3_3 = function_set['f3_3']
        P1 = np.array(P1)
        P2 = np.array(P2)
        P3 = np.array(P3)
        b = function_set['b']
        sgn_w0 = np.sign(w0)

            
        # Compute I_1
        I_1 = -abs(w0) * b + (dis_p['t1_0'] * f2_1) + (dis_p['t2_0'] * f2_2) + (dis_p['t3_0'] * f2_3)
        
        # Compute I_2
        I_2 = -np.dot(w_hat, sgn_w0 * b) - (np.dot(m1_hat, f2_1) + np.dot(m2_hat, f2_2) + np.dot(m3_hat, f2_3))
        
        # Compute the sum of mi_hat * f3i
        sum_mi_hat_f3i = (np.dot(m1_hat, f3_1) + 
                        np.dot(m2_hat, f3_2) + 
                        np.dot(m3_hat, f3_3))
        
        # Compute I_u_a and I_v_a
        I_u_a = 0.5 * np.dot(u_hat, sum_mi_hat_f3i)
        I_v_a = 0.5 * np.dot(v_hat, sum_mi_hat_f3i)
        
        # Compute I_u and I_v
        I_u = u0 * I_1 + I_u_a
        I_v = v0 * I_1 + I_v_a

        # Compute I_x, I_y and I_z
        I_x = u_hat[0] * I_u +  v_hat[0] * I_v + (P1[0] - P2[0]) * I_1
        I_y = u_hat[1] * I_u +  v_hat[1] * I_v + (P1[1] - P2[1]) * I_1
        I_z = u_hat[2] * I_u +  v_hat[2] * I_v + (P1[2] - P2[2]) * I_1 
        
        return {
            'I_1': I_1,
            'I_2': I_2,
            'I_u': I_u,
            'I_v': I_v,
            'I_x': I_x,
            'I_y': I_y,
            'I_z': I_z
        }
        
    # For sphere 560
    P1 = (0.891226087  ,  -0.406880893  ,  -0.200409581)
    P2 = (0.935016243  ,  -0.354604887  ,             0)
    P3 = (0.964457313  ,   -0.19224163  ,  -0.181287747)
    r =  (0.992708874  ,    0.12053668  ,             0) 

    # Compute triangle properties
    triangle_props = compute_triangle_properties(r,P1, P2, P3)

    # Compute s values for each side
    si_values = compute_si_values(triangle_props)

    # Compute distances to nodes from p
    dis_p = compute_distances_to_nodes(triangle_props)

    # Compute distance to nodes from r
    dis_r = compute_distances_to_node_from_r(triangle_props,si_values,dis_p)

    # Compute function set values
    function_set = compute_function_set(triangle_props, si_values, dis_r, dis_p)

    # Compute final I_1 value
    integrals = compute_I(triangle_props, function_set, dis_p, P1, P2, P3)
  

    return

quadrature_rules = {
        1: {
            'points': np.array([
                [0.333333333333333, 0.333333333333333]
            ]),
            'weights': np.array([0.5])
        },
        2: {
            'points': np.array([
                [0.166666666666667, 0.166666666666667],
                [0.666666666666667, 0.166666666666667],
                [0.166666666666667, 0.666666666666667]
            ]),
            'weights': np.array([0.166666666666667, 0.166666666666667, 0.166666666666667])
        },
        3: {
            'points': np.array([
                [0.333333333333333, 0.333333333333333], 
                [0.600000000000000, 0.200000000000000],
                [0.200000000000000, 0.600000000000000],
                [0.200000000000000, 0.200000000000000]
            ]),
            'weights': np.array([-0.28125, 0.2604166666666665,0.2604166666666665, 0.2604166666666665])
        },
        4: {
            'points': np.array([
                [0.108103018168070, 0.445948490915965], 
                [0.445948490915965, 0.108103018168070], 
                [0.445948490915965, 0.445948490915965],
                [0.816847572980459, 0.091576213509771], 
                [0.091576213509771, 0.816847572980459], 
                [0.091576213509771, 0.091576213509771]
            ]),
            'weights': np.array([
                0.1116907948390055, 0.1116907948390055, 0.1116907948390055,
                0.054975871827661, 0.054975871827661, 0.054975871827661
            ])
        },
        5: {
            'points': np.array([
                [0.333333333333333, 0.333333333333333],
                [0.059715871789770, 0.470142064105115], 
                [0.470142064105115, 0.059715871789770], 
                [0.470142064105115, 0.470142064105115],
                [0.797426985353087, 0.101286507323456], 
                [0.101286507323456, 0.797426985353087], 
                [0.101286507323456, 0.101286507323456]
            ]),
            'weights': np.array([
                0.1125, 0.066197076394253, 0.066197076394253, 0.066197076394253,
                0.0629695902724135, 0.0629695902724135, 0.0629695902724135
            ])
        },
        6: {
            'points': np.array([
                [0.501426509658179, 0.249286745170910], 
                [0.249286745170910, 0.501426509658179], 
                [0.249286745170910, 0.249286745170910],
                [0.873821971016996, 0.063089014491502], 
                [0.063089014491502, 0.873821971016996], 
                [0.063089014491502, 0.063089014491502],
                [0.053145049844817, 0.310352451033784], 
                [0.053145049844817, 0.636502499121399], 
                [0.310352451033784, 0.053145049844817], 
                [0.310352451033784, 0.636502499121399], 
                [0.636502499121399, 0.053145049844817], 
                [0.636502499121399, 0.310352451033784]
            ]),
            'weights': np.array([
                0.0583931378631895, 0.0583931378631895, 0.0583931378631895,
                0.0254224531851035, 0.0254224531851035, 0.0254224531851035,
                0.041425537809187, 0.041425537809187, 0.041425537809187, 
                0.041425537809187, 0.041425537809187, 0.041425537809187
            ])
        },
        7: {
            'points': np.array([
                [0.333333333333333, 0.333333333333333],
                [0.479308067841920, 0.260345966079040], 
                [0.260345966079040, 0.479308067841920], 
                [0.260345966079040, 0.260345966079040],
                [0.869739794195568, 0.065130102902216], 
                [0.065130102902216, 0.869739794195568], 
                [0.065130102902216, 0.065130102902216],
                [0.048690315425316, 0.312865496004874], 
                [0.048690315425316, 0.638444188569810], 
                [0.312865496004874, 0.048690315425316], 
                [0.312865496004874, 0.638444188569810], 
                [0.638444188569810, 0.048690315425316], 
                [0.638444188569810, 0.312865496004874]
            ]),
            'weights': np.array([
                -0.074785022233841, 0.087807628716604, 0.087807628716604, 0.087807628716604,
                0.026673617804419, 0.026673617804419, 0.026673617804419,
                0.038556880445129, 0.038556880445129, 0.038556880445129, 
                0.038556880445129, 0.038556880445129, 0.038556880445129
            ])
        }
    }


def read_custom_msh(file_path):
    with open(file_path, 'r') as file:
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
    
    return np.array(vertices), np.array(elements)

def find_internal_edges_and_third_vertices(vertices, elements):
    vertices_indices = elements
    edge_triangle_map = {}
    #Initialize vertex-to-triangle mapping
    vertex_to_triangles = defaultdict(set)
     
   # Create edge-to-triangle map
    for triangle_index, vertices in enumerate(vertices_indices):
        for i in range(3):
            vertex1 = vertices[i]
            vertex2 = vertices[(i + 1) % 3]
            edge = tuple(sorted([vertex1, vertex2]))
            if edge in edge_triangle_map:
                edge_triangle_map[edge].append(triangle_index + 1)
            else:
                edge_triangle_map[edge] = [triangle_index + 1]
            
            #Populate vertex-to-triangle mapping
            vertex_to_triangles[vertex1].add(triangle_index + 1)
 
    # Identify internal edges and third vertices
    internal_edges = []
    adjacent_cells = []
    edge_third_vertices = []
    edge_index = 1

    for edge, triangles in edge_triangle_map.items():
        if len(triangles) == 2:  # Check if exactly two triangles share this edge
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

    return internal_edges, adjacent_cells, edge_third_vertices, vertex_to_triangles


def build_neighborhood(elements, vertex_to_triangles):
    # Build neighborhood list for each triangle
    neighborhood = defaultdict(set)
    
    for triangle_index, vertices in enumerate(elements):
        triangle_id = triangle_index + 1
        for vertex in vertices:
            neighboring_triangles = vertex_to_triangles[vertex]
            for neighbor in neighboring_triangles:
                if neighbor != triangle_id:
                    neighborhood[triangle_id].add(neighbor)

        neighborhood[triangle_id].add(triangle_id)  # Ensure mutual inclusion    
    
    # Convert the sets to lists for final output
    neighborhood = {k: list(v) for k, v in neighborhood.items()}
    
    return neighborhood

def map_to_3d_triangle(vertices, s, t):
    # Extract vertices coordinates
    x1, y1, z1 = vertices[0]
    x2, y2, z2 = vertices[1]
    x3, y3, z3 = vertices[2]
    
    # Compute mapped point
    x_mapped = x1 + (x2 - x1) * s + (x3 - x1) * t
    y_mapped = y1 + (y2 - y1) * s + (y3 - y1) * t
    z_mapped = z1 + (z2 - z1) * s + (z3 - z1) * t
    
    mapped_point = np.array([x_mapped, y_mapped, z_mapped])
    
    return mapped_point

# Function to compute and store the mapped Gauss points for each triangle
def compute_mapped_gauss_points(vertices, elements, quadrature_rules, degree_m):
    mapped_gauss_points = {}
    
    for i, element in enumerate(elements):
        v1 = vertices[element[0] - 1]
        v2 = vertices[element[1] - 1]
        v3 = vertices[element[2] - 1]
        triangle_vertices = np.array([v1, v2, v3])
        
        gauss_points = []
        
        for s, t in quadrature_rules[degree_m]['points']:
            mapped_point = map_to_3d_triangle(triangle_vertices, s, t)
            gauss_points.append(mapped_point)
        
        mapped_gauss_points[i + 1] = np.array(gauss_points)
        
    return mapped_gauss_points

def calculate_basis_function(vertices, internal_edges, adjacent_cells, edge_third_vertices, mapped_gauss_points):
    basis_function_values = {}

    for edge_info in adjacent_cells:
        edge_index = edge_info[0]
        triangle_plus = edge_info[1]
        triangle_minus = edge_info[2]

        gauss_points_plus = mapped_gauss_points[triangle_plus]
        gauss_points_minus = mapped_gauss_points[triangle_minus]

        third_vertex_plus = vertices[edge_third_vertices[edge_index - 1][1] - 1]
        third_vertex_minus = vertices[edge_third_vertices[edge_index - 1][2] - 1]

        # Get vertices v1 and v2 for the current edge from internal_edges
        _, v1_idx, v2_idx = internal_edges[edge_index - 1]
        v1 = vertices[v1_idx - 1]
        v2 = vertices[v2_idx - 1]
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

        # Convert lists to NumPy arrays
        bf_values['p_plus'] = np.array(bf_values['p_plus'])
        bf_values['p_minus'] = np.array(bf_values['p_minus'])

        basis_function_values[edge_index] = bf_values

    return basis_function_values

def analytical_integral(point_m_plus,triangle_index):
    return    

def double_integral_for_impedence(green_function, bounded_green_function, f2, adjacent_cells, neighborhood, degree_m, degree_n, mapped_gauss_points, basis_function_values, medium_parameter, field_parameter):
    potential_values = {}
    impedence_matrix = {}
    #impedence_matrix = np.zeros((degree_m, degree_n), dtype=np.complex128)
    forcing_vector = {}

    # parameters for Green's function
    k = medium_parameter['wave_number']
    w = field_parameter['ang_frequency']
    mu_relative = medium_parameter['relative_permiability']

    #parameters for electric_field_incident
    K = field_parameter['propagation_vector']
    E_inc = field_parameter['E_incident']


    for edge_info in adjacent_cells:
        edge_index_m = edge_info[0]
        triangle_m_plus = edge_info[1]
        triangle_m_minus = edge_info[2]

        gauss_points_m_plus = mapped_gauss_points[triangle_m_plus]        
        gauss_points_m_minus = mapped_gauss_points[triangle_m_minus]
        gauss_weights_m = quadrature_rules[degree_m]['weights']

        lm = basis_function_values[edge_index_m]['length']
        p_m_plus = basis_function_values[edge_index_m]['p_plus']
        p_m_minus = basis_function_values[edge_index_m]['p_minus']
        
        for edge_info_n in adjacent_cells:
            edge_index_n = edge_info_n[0]
            triangle_n_plus = edge_info_n[1]
            triangle_n_minus = edge_info_n[2]

            gauss_points_n_plus = mapped_gauss_points[triangle_n_plus]
            gauss_points_n_minus = mapped_gauss_points[triangle_n_minus]
            gauss_weights_n = quadrature_rules[degree_n]['weights']

            ln = basis_function_values[edge_index_n]['length'] 
            p_n_plus = basis_function_values[edge_index_n]['p_plus']
            p_n_minus = basis_function_values[edge_index_n]['p_minus']
            
            f1 = green_function

            phi_mn_plus = 0
            A_mn_plus = 0
            phi_mn_minus = 0
            A_mn_minus = 0

            outer_sum_phi_mn = 0
            outer_sum_A_mn = 0
            sum_v_m = 0
    
            phi_mn_plus = 0
            A_mn_plus = 0
            phi_mn_minus = 0
            A_mn_minus = 0

            outer_sum_phi_mn = 0
            outer_sum_A_mn = 0
            sum_v_m = 0
         
            for i, point_m_plus in enumerate(gauss_points_m_plus):
                inner_sum_phi_m_plus = 0
                inner_sum_A_m_plus = 0
                
                if triangle_n_plus in neighborhood[triangle_m_plus]:
                    #be cautious of area term
                    inner_sum_m_plus = analytical_integral(mesh_basis_analysis, edge_index_n, triangle_n_plus, "plus", point_m_plus)                
                    for j, point_n_plus in enumerate(gauss_points_n_plus):
                        inner_sum_m_plus = gauss_weights_n[j] * bounded_green_function(point_m_plus, point_n_plus, k)
                        inner_sum_phi_m_plus += inner_sum_m_plus 
                        inner_sum_A_m_plus += p_n_plus[j] * inner_sum_m_plus

                else: 
                    inner_sum_m_plus = 0  
                    for j, point_n_plus in enumerate(gauss_points_n_plus):
                        inner_sum_m_plus = gauss_weights_n[j] * f1(point_m_plus, point_n_plus, k)
                        inner_sum_phi_m_plus += inner_sum_m_plus 
                        inner_sum_A_m_plus += p_n_plus[j] * inner_sum_m_plus
                
                if triangle_n_minus in neighborhood[triangle_m_plus]:
                    for j, point_n_minus in enumerate(gauss_points_n_minus):
                        inner_sum_m_plus = gauss_weights_n[j] * f1(point_m_plus, point_n_minus, k)
                        inner_sum_phi_m_plus -= inner_sum_m_plus # grad of f_n for r' in T_minus is negative.
                        inner_sum_A_m_plus += p_n_minus[j] * inner_sum_m_plus

                else:
                    for j, point_n_minus in enumerate(gauss_points_n_minus):
                        inner_sum_m_plus = gauss_weights_n[j] * f1(point_m_plus, point_n_minus, k)
                        inner_sum_phi_m_plus -= inner_sum_m_plus # grad of f_n for r' in T_minus is negative.
                        inner_sum_A_m_plus += p_n_minus[j] * inner_sum_m_plus

                phi_mn_plus = -ln / (4 * np.pi * 1j * k) * inner_sum_phi_m_plus # The plus is over m only in mn_plus 
                A_mn_plus = (ln * 5e-8 * mu_relative) * inner_sum_A_m_plus

                outer_sum_phi_mn -= gauss_weights_m[i] * phi_mn_plus
                outer_sum_A_mn += np.dot(p_m_plus[i], (gauss_weights_m[i] * A_mn_plus))

                E_m_plus = f2(E_inc, K, point_m_plus)
                sum_v_m += np.dot(p_m_plus[i], (gauss_weights_m[i] * E_m_plus))

            for i, point_m_minus in enumerate(gauss_points_m_minus):
                inner_sum_phi_m_minus = 0
                inner_sum_A_m_minus = 0

                for j, point_n_plus in enumerate(gauss_points_n_plus):
                    inner_sum_m_minus = gauss_weights_n[j] * f1(point_m_minus, point_n_plus, k)
                    inner_sum_phi_m_minus += inner_sum_m_minus 
                    inner_sum_A_m_minus += p_n_plus[j] * inner_sum_m_minus

                for j, point_n_minus in enumerate(gauss_points_n_minus):
                    inner_sum_m_minus = gauss_weights_n[j] * f1(point_m_minus, point_n_minus, k)
                    inner_sum_phi_m_minus -= inner_sum_m_minus 
                    inner_sum_A_m_minus += p_n_minus[j] * inner_sum_m_minus
                
                phi_mn_minus = (-ln / (4 * np.pi * 1j * k)) * inner_sum_phi_m_minus # The minus is over m only in mn_plus 
                A_mn_minus = (ln * 5e-8 * mu_relative) * inner_sum_A_m_minus

                outer_sum_phi_mn += gauss_weights_m[i] * phi_mn_minus 
                outer_sum_A_mn += np.dot(p_m_minus[i], (gauss_weights_m[i] * A_mn_minus))
                
                E_m_minus = f2(E_inc, K, point_m_minus)
                sum_v_m += np.dot(p_m_minus[i], (gauss_weights_m[i] * E_m_minus))

            Z_mn = lm * (0.5j * w * outer_sum_A_mn + outer_sum_phi_mn)
            V_m  = lm * sum_v_m    
            # Store the values in potential_values
            potential_values[(edge_index_m, edge_index_n)] = {
                'A_mn': [A_mn_plus, A_mn_minus],
                'phi_mn': [phi_mn_plus, phi_mn_minus]
            }
            # Store the values in impedence_matrix
            impedence_matrix[(edge_index_m, edge_index_n)] = Z_mn 
            #impedence_matrix[edge_index_m - 1, edge_index_n - 1] = Z_mn
            # Store the values in forcing_vector
            forcing_vector[edge_index_m] = V_m

            # Print A_mn for the current pair of edges
            #print(f"Potential values for edge pair ({edge_index_m}, {edge_index_n}):")
            #print("A_mn:", potential_values[(edge_index_m, edge_index_n)]['A_mn'])
            #print("phi_mn:", potential_values[(edge_index_m, edge_index_n)]['phi_mn'])
            #print(f"Z_mn values for edge pair ({edge_index_m}, {edge_index_n}):")
            #print(impedence_matrix[(edge_index_m, edge_index_n)])
        #print(f"V_m values for edge pair ({edge_index_m}):")
        #print("V_m:",forcing_vector[edge_index_m])
            

    # Print p_plus values for all edges
    #for edge_index, values in basis_function_values.items():
    #    print(f"p_plus for edge {edge_index}:")
    #    print(values['p_plus'])

    return potential_values,impedence_matrix,forcing_vector

def green_function(mapped_point_m, mapped_point_n, wave_number):
    k= wave_number
    R = np.linalg.norm(mapped_point_m - mapped_point_n)
    return (np.exp(-1j * k * R)) / R

def electric_field_incident(E_incident, propagation_vector, mapped_point_m):
    E_inc = E_incident
    K = propagation_vector
    phase = np.dot(K,mapped_point_m)
    return E_inc * np.exp(-1j * phase)

def bounded_green_function(mapped_point_m, mapped_point_n, wave_number):
    k= wave_number
    R = np.linalg.norm(mapped_point_m - mapped_point_n)
    return (np.exp(-1j * k * R) - 1) / R


def main():

    # Path to your .msh file
    file_path = 'E:\MNIT_Internship\Sphere560.msh'

    # Read the custom .msh file
    vertices, elements = read_custom_msh(file_path)

    # Process the mesh data
    internal_edges, adjacent_cells, edge_third_vertices, vertex_to_triangles = find_internal_edges_and_third_vertices(vertices, elements)
    neighborhood = build_neighborhood(elements, vertex_to_triangles)

    # Select the degree of the Gaussian quadrature rule for (Test Procedure / Observation points / m triangles)
    degree_m = int(input("Enter the degree_1 of the Gaussian quadrature rule (1-7) for Observation poitns : "))

    # Select the degree of the Gaussian quadrature rule for (Green Function / Source points / n triangles)
    degree_n = degree_m 
    #degree_n = int(input("Enter the degree_1 of the Gaussian quadrature rule (1-7) for Source poitns : "))
    
    #
    mapped_gauss_points = compute_mapped_gauss_points(vertices, elements, quadrature_rules, degree_m)


    # Calculate the basis function values 
    basis_function_values = calculate_basis_function(vertices, internal_edges, adjacent_cells, edge_third_vertices, mapped_gauss_points)
    
    # Print the results
    print("Basis Function Values at Mapped Gauss Points:") 
    
    
    # Start measuring time
    start_time = time.time()
    
    w = 1
    [k_x , k_y , k_z] = [1,0,0]
    K = np.array([k_x , k_y , k_z]) # Captital K is used
    [E_x , E_y , E_z] = [1,0,0]
    E_inc = np.array([E_x , E_y , E_z])
    field_parameter = {'ang_frequency' : w ,'propagation_vector': K ,'E_incident':E_inc}

    mu_relative = 1
    epsilon_relative = 1
    k = w * np.sqrt(mu_relative * epsilon_relative) * 3.33e-9
    medium_parameter = {'relative_permiability': mu_relative ,'relative_permitivity': epsilon_relative ,'wave_number' : k}

    # Calculate the integral
    potential_values,impedence_matrix,forcing_vector = double_integral_for_impedence(green_function, bounded_green_function, electric_field_incident, adjacent_cells, neighborhood, degree_m, degree_n, mapped_gauss_points, basis_function_values, medium_parameter, field_parameter)

    # End measuring time
    end_time = time.time()

    print(forcing_vector)
    # Calculate and print the time taken
    execution_time = end_time - start_time
    print("Time taken:", execution_time, "seconds")


    '''
# Example impedance matrix Z (3x3 matrix)
Z = np.array([[1.0, 0.1, 0.2],
              [0.1, 1.5, 0.3],
              [0.2, 0.3, 1.8]])

# Example voltage vector V (3x1 vector)
V = np.array([5.0, 10.0, 15.0])

# Solve for the current vector I using CGS method
I, info = cgs(impedence_matrix, forcing_vector)

if info == 0:
    print("Converged successfully")
elif info > 0:
    print(f"Converged in {info} iterations")
else:
    print("Failed to converge")

print("Current vector I:")
print(I)



    # Select the function to integrate
    print("Select the function to integrate:")
    print("1. f(x, y) = psi_mn")
    print("2. f(x, y) = A_mn")
    choice = int(input("Enter your choice (1 or 2): "))

    if choice == 1:
        integral_result = (Lm*ln)* gaussian_quadrature_triangular(test_procedure1, degree, vertices) # vertices of edge m=1 and n= 2 
        print(f"Integral of I_1 over the triangular domain: {integral_result}")
    elif choice == 2:
        
    else:
        print("Invalid choice. Please enter 1 or 2.")


'''


if __name__ == "__main__":
    main()
