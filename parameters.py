class Parameters:
    def __init__(self, medium_parameter, field_parameter):
        self.medium_parameter = medium_parameter
        self.field_parameter = field_parameter
    def __init__(self, wave_number, relative_permeability, ang_frequency, E_incident, propagation_vector):
        self.wave_number = wave_number
        self.relative_permeability = relative_permeability
        self.ang_frequency = ang_frequency
        self.E_incident = E_incident
        self.propagation_vector = propagation_vector    
    def get_wave_number(self):
        return self.medium_parameter['wave_number']

    def get_ang_frequency(self):
        return self.field_parameter['ang_frequency']

    def get_relative_permeability(self):
        return self.medium_parameter['relative_permiability']

    def get_propagation_vector(self):
        return self.field_parameter['propagation_vector']

    def get_e_incident(self):
        return self.field_parameter['E_incident']
