import pandas as pd
import json
import os
import numpy as np

# Base scene to be extended

class BaseScene:

    def __init__(self, name: str, components: dict):
        self.name = name
        self.components = components

    def initialize(self):
        raise NotImplementedError

    def evaluate(self, x):
        raise NotImplementedError

    def repair(self, x):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

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

                


