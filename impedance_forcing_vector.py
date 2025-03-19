import numpy as np
from analytical_integration import Analytical_Integration
class Impedance_Forcing_Vector:
    def __init__(self, mesh_basis_analysis, degree_m, degree_n, quadrature_rules, medium_parameter, field_parameter):
        self.mesh_basis_analysis = mesh_basis_analysis
        self.adjacent_cells = mesh_basis_analysis.adjacent_cells
        self.neighborhood = mesh_basis_analysis.neighborhood
        self.quadrature_rules = quadrature_rules
        self.degree_m = degree_m
        self.degree_n = degree_n
        self.mapped_gauss_points = mesh_basis_analysis.mapped_gauss_points
        self.basis_function_values = mesh_basis_analysis.basis_function_values
        self.medium_parameter = medium_parameter
        self.field_parameter = field_parameter

    def integrate(self):
        potential_values = {}
        #impedance_matrix = {}
        #forcing_vector = {}
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


        for edge_info in self.adjacent_cells:
            edge_index_m = edge_info[0]
            triangle_m_plus = edge_info[1]
            triangle_m_minus = edge_info[2]
            #print(f"Edge m:{edge_index_m} exicuted with tm+:{triangle_m_plus} and tm-:{triangle_m_minus}")

            gauss_points_m_plus = self.mapped_gauss_points[triangle_m_plus]        
            gauss_points_m_minus = self.mapped_gauss_points[triangle_m_minus]
            gauss_weights_m = self.quadrature_rules[self.degree_m]['weights']

            lm = self.basis_function_values[edge_index_m]['length']
            p_m_plus = self.basis_function_values[edge_index_m]['p_plus']
            p_m_minus = self.basis_function_values[edge_index_m]['p_minus']
            
            for edge_info_n in self.adjacent_cells:
                edge_index_n = edge_info_n[0]
                triangle_n_plus = edge_info_n[1]
                triangle_n_minus = edge_info_n[2]
                #print(f"Edge n:{edge_index_n} exicuted with tn+:{triangle_n_plus} and tn-:{triangle_n_minus}")

                gauss_points_n_plus = self.mapped_gauss_points[triangle_n_plus]
                gauss_points_n_minus = self.mapped_gauss_points[triangle_n_minus]
                gauss_weights_n = self.quadrature_rules[self.degree_n]['weights']

                ln = self.basis_function_values[edge_index_n]['length'] 
                p_n_plus = self.basis_function_values[edge_index_n]['p_plus']
                p_n_minus = self.basis_function_values[edge_index_n]['p_minus']

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
                    
                    if triangle_n_plus in self.neighborhood[triangle_m_plus]:
                        # Use the AnalyticalIntegration instance method
                        #print(f"executed {triangle_m_plus},{triangle_n_plus}, plus analytical function")
                        ai = Analytical_Integration(self.mesh_basis_analysis, edge_index_n, triangle_n_plus, "plus", point_m_plus)
                        ai_result = ai.analytical_integration()
                        inner_sum_phi_m_plus, inner_sum_A_m_plus = ai_result['I_1'], ai_result['I_A_m']  
                        for j, point_n_plus in enumerate(gauss_points_n_plus):
                            inner_sum_m_plus = gauss_weights_n[j] * bounded_green_function(point_m_plus, point_n_plus, k)
                            inner_sum_phi_m_plus += inner_sum_m_plus 
                            inner_sum_A_m_plus += p_n_plus[j] * inner_sum_m_plus
                    else:  
                        for j, point_n_plus in enumerate(gauss_points_n_plus):
                            inner_sum_m_plus = gauss_weights_n[j] * green_function(point_m_plus, point_n_plus, k)
                            inner_sum_phi_m_plus += inner_sum_m_plus 
                            inner_sum_A_m_plus += p_n_plus[j] * inner_sum_m_plus                   
                    
                    if triangle_n_minus in self.neighborhood[triangle_m_plus]:
                        #print(f"executed {triangle_m_plus},{triangle_n_minus} minus analytical function")
                        ai = Analytical_Integration(self.mesh_basis_analysis, edge_index_n, triangle_n_minus, "minus", point_m_plus)
                        ai_result = ai.analytical_integration()
                        inner_sum_phi_m_plus, inner_sum_A_m_plus = ai_result['I_1'], ai_result['I_A_m'] 
                        for j, point_n_minus in enumerate(gauss_points_n_minus):
                            inner_sum_m_plus = gauss_weights_n[j] * bounded_green_function(point_m_plus, point_n_minus, k)
                            inner_sum_phi_m_plus -= inner_sum_m_plus # grad of f_n for r' in T_minus is negative.
                            inner_sum_A_m_plus += p_n_minus[j] * inner_sum_m_plus         
                    else:
                        for j, point_n_minus in enumerate(gauss_points_n_minus):
                            inner_sum_m_plus = gauss_weights_n[j] * green_function(point_m_plus, point_n_minus, k)
                            inner_sum_phi_m_plus -= inner_sum_m_plus # grad of f_n for r' in T_minus is negative.
                            inner_sum_A_m_plus += p_n_minus[j] * inner_sum_m_plus

                    phi_mn_plus = -(ln / (4 * np.pi * 1j * w * 8.85e-12)) * inner_sum_phi_m_plus # The plus is over m only in mn_plus 
                    A_mn_plus = (ln * 5e-8 * mu_relative) * inner_sum_A_m_plus
                    #A_mn_plus = (ln * 0.5 * k) * inner_sum_A_m_plus

                    outer_sum_phi_mn -= gauss_weights_m[i] * phi_mn_plus
                    outer_sum_A_mn += np.dot(p_m_plus[i], (gauss_weights_m[i] * A_mn_plus))

                    E_m_plus = electric_field_incident(E_inc, K, point_m_plus)
                    sum_v_m += np.dot(p_m_plus[i], (gauss_weights_m[i] * E_m_plus))

                for i, point_m_minus in enumerate(gauss_points_m_minus):
                    inner_sum_phi_m_minus = 0
                    inner_sum_A_m_minus = 0
                    
                    if triangle_n_plus in self.neighborhood[triangle_m_minus]:
                        #print(f"executed {triangle_m_minus},{triangle_n_plus} plus analytical function")
                        ai = Analytical_Integration(self.mesh_basis_analysis, edge_index_n, triangle_n_plus, "plus", point_m_minus)
                        ai_result = ai.analytical_integration()
                        inner_sum_phi_m_minus, inner_sum_A_m_minus = ai_result['I_1'], ai_result['I_A_m'] 
                        for j, point_n_plus in enumerate(gauss_points_n_plus):
                            inner_sum_m_minus = gauss_weights_n[j] * bounded_green_function(point_m_minus, point_n_plus, k)
                            inner_sum_phi_m_minus += inner_sum_m_minus 
                            inner_sum_A_m_minus += p_n_plus[j] * inner_sum_m_minus
                    else: 
                        for j, point_n_plus in enumerate(gauss_points_n_plus):
                            inner_sum_m_minus = gauss_weights_n[j] * green_function(point_m_minus, point_n_plus, k)
                            inner_sum_phi_m_minus += inner_sum_m_minus 
                            inner_sum_A_m_minus += p_n_plus[j] * inner_sum_m_minus
                    
                    if triangle_n_minus in self.neighborhood[triangle_m_minus]:
                        #print(f"executed {triangle_m_minus},{triangle_n_minus} minus analytical function")
                        ai = Analytical_Integration(self.mesh_basis_analysis, edge_index_n, triangle_n_minus, "minus", point_m_minus)
                        ai_result = ai.analytical_integration()
                        inner_sum_phi_m_minus, inner_sum_A_m_minus = ai_result['I_1'], ai_result['I_A_m'] 
                        for j, point_n_minus in enumerate(gauss_points_n_minus):
                            inner_sum_m_minus = gauss_weights_n[j] * bounded_green_function(point_m_minus, point_n_minus, k)
                            inner_sum_phi_m_minus -= inner_sum_m_minus 
                            inner_sum_A_m_minus += p_n_minus[j] * inner_sum_m_minus
                    else: 
                        for j, point_n_minus in enumerate(gauss_points_n_minus):
                            inner_sum_m_minus = gauss_weights_n[j] * green_function(point_m_minus, point_n_minus, k)
                            inner_sum_phi_m_minus -= inner_sum_m_minus 
                            inner_sum_A_m_minus += p_n_minus[j] * inner_sum_m_minus                   
                    
                    phi_mn_minus = (-ln / (4 * np.pi * 1j * w * 8.85e-12 )) * inner_sum_phi_m_minus # The minus is over m only in mn_plus 
                    A_mn_minus = (ln * 5e-8 * mu_relative) * inner_sum_A_m_minus
                    #A_mn_minus = (ln * 0.5 * k) * inner_sum_A_m_minus
                    
                    outer_sum_phi_mn += gauss_weights_m[i] * phi_mn_minus 
                    outer_sum_A_mn += np.dot(p_m_minus[i], (gauss_weights_m[i] * A_mn_minus))
                    
                    E_m_minus = electric_field_incident(E_inc, K, point_m_minus)
                    sum_v_m += np.dot(p_m_minus[i], (gauss_weights_m[i] * E_m_minus))

                Z_mn = lm * (0.5j * w * outer_sum_A_mn + outer_sum_phi_mn) # w is not written here instead its absorbed into k term in A_mn_
                V_m  = lm * sum_v_m    
                # Store the values in potential_values
                potential_values[(edge_index_m, edge_index_n)] = {
                    'A_mn': [A_mn_plus, A_mn_minus],
                    'phi_mn': [phi_mn_plus, phi_mn_minus]
                }
                # Store the values in impedance_matrix
                #impedance_matrix[(edge_index_m, edge_index_n)] = Z_mn 
                # Store the values in forcing_vector
                #forcing_vector[edge_index_m] = V_m

                impedance_matrix[edge_index_m - 1, edge_index_n - 1] = Z_mn 
                forcing_vector[edge_index_m - 1] = V_m

         
            #print(f"edge_index_m: {edge_index_m} executed")
        return potential_values,impedance_matrix,forcing_vector
    
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
    if R == 0:
        return 0
    else:
        return (np.exp(-1j * k * R) - 1) / R

