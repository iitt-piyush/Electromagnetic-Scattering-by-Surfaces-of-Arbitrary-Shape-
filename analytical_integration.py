import numpy as np

class Analytical_Integration:
    def __init__(self, mesh_basis_analysis, edge_index, triangle_index, triangle_sign, observation_point_r):
        self.mesh_basis_analysis = mesh_basis_analysis
        self.vertices = mesh_basis_analysis.vertices
        self.elements = mesh_basis_analysis.elements
        self.edge_third_vertices = mesh_basis_analysis.edge_third_vertices
        self.edge_index = edge_index
        self.triangle_index = triangle_index
        self.triangle_sign = triangle_sign
        self.observation_point_r = observation_point_r

    def compute_triangle_properties(self, r, P1, P2, P3):
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
        
        return {
            'side_lengths': {'l1': l1, 'l2': l2, 'l3': l3},
            'normal_vector': n_hat,
            'area': A,
            'unit_vectors': {'u_hat': u_hat, 'v_hat': v_hat, 'w_hat': w_hat},
            'side_vectors': {'s1_hat': s1_hat, 's2_hat': s2_hat, 's3_hat': s3_hat, 'm1_hat': m1_hat, 'm2_hat': m2_hat, 'm3_hat': m3_hat},
            'local_coordinates_P3': {'u3': u3, 'v3': v3},
            'local_coordinates_r':  {'u0': u0, 'v0': v0, 'w0':w0}
        }

    def compute_si_values(self, triangle_props):

        # Extract triangle properties
        l1, l2, l3 = triangle_props['side_lengths'].values()
        u3, v3 = triangle_props['local_coordinates_P3'].values()
        u0, v0, _ = triangle_props['local_coordinates_r'].values()
        
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
    def compute_distances_to_nodes(self, triangle_props):

        # Extract triangle properties
        l1, l2, l3 = triangle_props['side_lengths'].values()
        u3, v3 = triangle_props['local_coordinates_P3'].values()
        u0, v0, _ = triangle_props['local_coordinates_r'].values()

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

    def compute_distances_to_node_from_r(self, triangle_props, si_values, dis_p):
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

    def compute_f2i(self, Ri_plus, Ri_minus, si_plus, si_minus):
        f2i = np.log((Ri_plus + si_plus) / (Ri_minus + si_minus))
        return np.nan_to_num(f2i, nan=0.0, posinf=0.0, neginf=0.0)

    def compute_f3i(self, si_plus, si_minus, Ri_plus, Ri_minus, Ri_0, f2i):
        f3i = (si_plus * Ri_plus - si_minus * Ri_minus) + (Ri_0**2 * f2i)
        return np.nan_to_num(f3i, nan=0.0, posinf=0.0, neginf=0.0)

    def compute_bi(self, ti_0, si_plus, si_minus, Ri_plus, Ri_minus, Ri_0, w0):
        term1 = np.arctan(ti_0 * si_plus / (Ri_0**2 + np.abs(w0) * Ri_plus))
        term2 = np.arctan(ti_0 * si_minus / (Ri_0**2 + np.abs(w0) * Ri_minus))
        bi = term1 - term2
        return np.nan_to_num(bi, nan=0.0, posinf=0.0, neginf=0.0)

    def compute_function_set(self, triangle_props, si_values, dis_r, dis_p):
        # Extract necessary values
        w0 = triangle_props['local_coordinates_r']['w0']
        
        # Compute f2i for each side
        f2_1 = self.compute_f2i(dis_r['R1_plus'], dis_r['R1_minus'], si_values['s1_plus'], si_values['s1_minus'])
        f2_2 = self.compute_f2i(dis_r['R2_plus'], dis_r['R2_minus'], si_values['s2_plus'], si_values['s2_minus'])
        f2_3 = self.compute_f2i(dis_r['R3_plus'], dis_r['R3_minus'], si_values['s3_plus'], si_values['s3_minus'])
        
        # Compute f3i for each side
        f3_1 = self.compute_f3i(si_values['s1_plus'], si_values['s1_minus'], dis_r['R1_plus'], dis_r['R1_minus'], dis_r['R1_0'], f2_1)
        f3_2 = self.compute_f3i(si_values['s2_plus'], si_values['s2_minus'], dis_r['R2_plus'], dis_r['R2_minus'], dis_r['R2_0'], f2_2)
        f3_3 = self.compute_f3i(si_values['s3_plus'], si_values['s3_minus'], dis_r['R3_plus'], dis_r['R3_minus'], dis_r['R3_0'], f2_3)
        
        # Compute bi for each side compute_bi(ti0, si_plus, si_minus, Ri_plus, Ri_minus, R0, w0):
        b1 = self.compute_bi(dis_p['t1_0'], si_values['s1_plus'], si_values['s1_minus'], dis_r['R1_plus'], dis_r['R1_minus'], dis_r['R1_0'], w0)
        b2 = self.compute_bi(dis_p['t2_0'], si_values['s2_plus'], si_values['s2_minus'], dis_r['R2_plus'], dis_r['R2_minus'], dis_r['R2_0'], w0)
        b3 = self.compute_bi(dis_p['t3_0'], si_values['s3_plus'], si_values['s3_minus'], dis_r['R3_plus'], dis_r['R3_minus'], dis_r['R3_0'], w0)
        
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

    def compute_I(self, triangle_props, function_set, dis_p, P1, P2, P3):
        # Extract necessary values
        w0 = triangle_props['local_coordinates_r']['w0']
        u_hat, v_hat, w_hat  = triangle_props['unit_vectors'].values()
        m1_hat = triangle_props['side_vectors']['m1_hat']
        m2_hat = triangle_props['side_vectors']['m2_hat']
        m3_hat = triangle_props['side_vectors']['m3_hat']
        area = triangle_props['area']
        u0, v0, _ = triangle_props['local_coordinates_r'].values()
        f2_1 = function_set['f2_1']
        f2_2 = function_set['f2_2']
        f2_3 = function_set['f2_3']
        f3_1 = function_set['f3_1']
        f3_2 = function_set['f3_2']
        f3_3 = function_set['f3_3']
        P1 = np.array(P1)

        b = function_set['b']
        sgn_w0 = np.sign(w0)
        
        # Compute I_1
        I_1 = ((-abs(w0) * b + (dis_p['t1_0'] * f2_1) + (dis_p['t2_0'] * f2_2) + (dis_p['t3_0'] * f2_3)) + 0j) / area
        
        # Compute I_2
        #I_2 = -np.dot(w_hat, sgn_w0 * b) - (np.dot(m1_hat, f2_1) + np.dot(m2_hat, f2_2) + np.dot(m3_hat, f2_3))
        
        # Compute the sum of mi_hat * f3i
        sum_mi_hat_f3i = (np.dot(m1_hat, f3_1) + 
                        np.dot(m2_hat, f3_2) + 
                        np.dot(m3_hat, f3_3))
        
        # Compute I_u_a and I_v_a
        I_u_a = 0.5 * (np.dot(u_hat, sum_mi_hat_f3i) + 0j) / area
        I_v_a = 0.5 * (np.dot(v_hat, sum_mi_hat_f3i) + 0j) / area
        
        # Compute I_u and I_v
        I_u = u0 * I_1 + I_u_a
        I_v = v0 * I_1 + I_v_a

        third_vertex_plus = self.vertices[self.edge_third_vertices[self.edge_index - 1][1] - 1]
        third_vertex_minus = self.vertices[self.edge_third_vertices[self.edge_index - 1][2] - 1]
        if self.triangle_sign == "plus":
            #print("analytic plus executed ")
            # Compute I_x, I_y and I_z
            I_x = u_hat[0] * I_u +  v_hat[0] * I_v + (P1[0] - third_vertex_plus[0]) * I_1
            I_y = u_hat[1] * I_u +  v_hat[1] * I_v + (P1[1] - third_vertex_plus[1]) * I_1
            I_z = u_hat[2] * I_u +  v_hat[2] * I_v + (P1[2] - third_vertex_plus[2]) * I_1 
        
        if self.triangle_sign == "minus":
            #print("analytic minus executed ")
            # Compute I_x, I_y and I_z
            I_x = u_hat[0] * I_u +  v_hat[0] * I_v + (third_vertex_minus[0] - P1[0]) * I_1 
            I_y = u_hat[1] * I_u +  v_hat[1] * I_v + (third_vertex_minus[1] - P1[1]) * I_1
            I_z = u_hat[2] * I_u +  v_hat[2] * I_v + (third_vertex_minus[2] - P1[2]) * I_1 
        
        return {
            'I_1': I_1,
            #'I_2': I_2,
            'I_u': I_u,
            'I_v': I_v,
            'I_A_m': np.array((I_x,I_y,I_z))
            }

    def analytical_integration(self):
        P1, P2, P3 = self.vertices[self.elements[self.triangle_index - 1] - np.array([1, 1, 1])]
        triangle_props = self.compute_triangle_properties(self.observation_point_r, P1, P2, P3)
        si_values = self.compute_si_values(triangle_props)
        dis_p = self.compute_distances_to_nodes(triangle_props)
        dis_r = self.compute_distances_to_node_from_r(triangle_props, si_values, dis_p)
        function_set = self.compute_function_set(triangle_props, si_values, dis_r, dis_p)
        integrals = self.compute_I(triangle_props, function_set, dis_p, P1, P2, P3)
        return integrals
 