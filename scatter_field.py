import numpy as np
class Scatter_Field:
    def __init__(self, mesh_basis_analysis, degree_m, degree_n, quadrature_rules, medium_parameter, field_parameter, I, observation_point):
        self.mesh_basis_analysis = mesh_basis_analysis
        self.adjacent_cells = mesh_basis_analysis.adjacent_cells
        self.quadrature_rules = quadrature_rules
        self.degree_m = degree_m
        self.degree_n = degree_n
        self.mapped_gauss_points = mesh_basis_analysis.mapped_gauss_points
        self.basis_function_values = mesh_basis_analysis.basis_function_values
        self.medium_parameter = medium_parameter
        self.field_parameter = field_parameter
        self.I = I
        self.observation_point = observation_point

    def integrate(self):
        potential_values = {}
        num_nodes = len(self.adjacent_cells)
        impedance_matrix = np.zeros((num_nodes, num_nodes), dtype=np.complex128)
        forcing_vector = np.zeros(num_nodes, dtype=np.complex128)


        # parameters for Green's function
        k = self.medium_parameter['wave_number']
        w = self.field_parameter['ang_frequency']
        mu_relative = self.medium_parameter['relative_permiability']

        #parameters for electric_field_incident
        K = self.field_parameter['propagation_vector']
        E_inc = self.field_parameter['E_incident']     
        i=0
        for edge_info_n in self.adjacent_cells:
            edge_index_n = edge_info_n[0]
            triangle_n_plus = edge_info_n[1]
            triangle_n_minus = edge_info_n[2]
            #print(f"Edge n:{edge_index_n} exicuted with tn+:{triangle_n_plus} and tn-:{triangle_n_minus}")

            gauss_points_n_plus = self.mapped_gauss_points[triangle_n_plus]
            gauss_points_n_minus = self.mapped_gauss_points[triangle_n_minus]
            gauss_weights_n = self.quadrature_rules[self.degree_n]['weights']

            ln = self.basis_function_values[edge_index_n]['length']
            I = self.I 
            p_n_plus = self.basis_function_values[edge_index_n]['p_plus']
            p_n_minus = self.basis_function_values[edge_index_n]['p_minus']

            phi_n = 0
            A_n = 0

            sum_phi_n = 0 
            sum_A_n = 0                  
            for j, point_n_plus in enumerate(gauss_points_n_plus):
                sum_phi_n += gauss_weights_n[j] * grad_green_function(self.observation_point, point_n_plus, k)
                sum_A_n += p_n_plus[j] * gauss_weights_n[j] * green_function(self.observation_point, point_n_plus, k)                   
            
            for j, point_n_minus in enumerate(gauss_points_n_minus):
                sum_phi_n -= gauss_weights_n[j] * grad_green_function(self.observation_point, point_n_minus, k) # grad of f_n for r' in T_minus is negative.
                sum_A_n += p_n_minus[j] * gauss_weights_n[j] * green_function(self.observation_point, point_n_minus, k)

            phi_n += ln * I[i] * sum_phi_n 
            A_n += ln * I[i] * sum_A_n 
            i= i + 1
            #E_m_plus = electric_field_incident(E_inc, K, self.observation_point)

        E_scat = -1 * (1j * w * 5e-8 * mu_relative * A_n +  (1 / (4 * np.pi * 1j * w * 8.85e-12 )) * phi_n)
        
        #print(f"edge_index_m: {edge_index_m} executed")
        return E_scat
    
def green_function(observation_point, mapped_point_n, wave_number):
    k= wave_number
    R = np.linalg.norm(observation_point - mapped_point_n)
    return (np.exp(-1j * k * R)) / R
def grad_green_function(observation_point, mapped_point_n, wave_number):
    k= wave_number
    R = np.linalg.norm(observation_point - mapped_point_n)
    return (np.dot( (observation_point-mapped_point_n), (np.exp(-1j * k * R) / R**2) ) * (-1j * k + (1 / R)))
def electric_field_incident(E_incident, propagation_vector, mapped_point_m):
    E_inc = E_incident
    K = propagation_vector
    phase = np.dot(K,mapped_point_m)
    return E_inc * np.exp(-1j * phase)



