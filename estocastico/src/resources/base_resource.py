# Defines the base resource class for all resources
# Every resource should inherit from this class
# Has the following properties:
#   - name: str
#   - value: np.array
#   - lb: np.array
#   - ub: np.array
#   - cost: np.array

import numpy as np
import json
import os
import pandas as pd

class BaseResource:
    """
    Base class for all resources.
    Name, value, lb, ub, and cost are required.
    Name: str
    Value: NumPy array
    Lower bound: NumPy array
    Upper bound: NumPy array
    Cost: NumPy array
    """

    def __init__(self,
                 name: str,
                 value: np.array,
                 lower_bound: np.array,
                 upper_bound: np.array,
                 cost: np.array):
        self.name = name
        self.value = value
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.cost = cost

    def __repr__(self):
        return f'{self.name}'

    def __str__(self):
        return f'{self.name}'

    def __hash__(self):
        return hash(self.name)

    def __len__(self):
        return len(self.value)

    def __contains__(self, item):
        return item in self.value

    def __add__(self, other):
        return self.value + other.value

    def __sub__(self, other):
        return self.value - other.value

    def __mul__(self, other):
        return self.value * other.value

    def __truediv__(self, other):
        return self.value / other.value

    def __floordiv__(self, other):
        return self.value // other.value

    def __mod__(self, other):
        return self.value % other.value

    def __divmod__(self, other):
        return divmod(self.value, other.value)

    def __pow__(self, other):
        return self.value ** other.value

    def ravel(self):
        return self.value.ravel()

    def shape(self):
        return self.value.shape

    # @TODO ADDED BY LARISSA:
    def to_json(self, filename=None, path=None, json_= True, excel_=True):

        # Getting the attributes - It is a dict object:
        attributes_np = vars(self)
        
        attributes_dict = {}
        # Getting the class name
        classname = self.__class__.__name__ 

        # Converting the value of the dict attributes from np.ndarray to list
        for key, value in attributes_np.items():
            if isinstance(value, np.ndarray):
                attributes_dict[key] = value.tolist()
        
        # Converting to JSON
        json_data = json.dumps(attributes_dict, indent=4)

        if not filename:
            filename = classname
            
        if path:
            if not os.path.exists(path):
                os.makedirs(path)
            filename = path + "/" + filename

        json_filename = filename + ".json"
        with open(json_filename, 'w') as json_file:
            json_file.write(json_data)

        if excel_:                        
            #Cria um dicionário de DataFrames a partir do dicionário de arrays 2D
            dfs_ = {_key: pd.DataFrame(_value) for _key, _value in attributes_dict.items()}
            
            # Renomear as colunas em cada DataFrame
            for df_ in dfs_.values():
                df_.columns = [f'Item {i}' for i in range(df_.shape[1])]
            
            # Nome do arquivo Excel
            excel_filename = filename + ".xlsx"
            
            # Salvar cada DataFrame em uma folha diferente
            with pd.ExcelWriter(excel_filename) as writer:
                for key_, _df in dfs_.items():  # Use dfs_ e não attributes_dict
                    _df.to_excel(writer, sheet_name=key_, index=False)  # Salva cada DataFrame na folha correspondente

                

