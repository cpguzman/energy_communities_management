import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# Base scene to be extended

class BaseScene:

    def __init__(self, name: str, components: dict):
        self.name = name
        self.components = components
        self.project_colors = {"green": "#58c1ae", "blue": "#0e73b9", "lightgreen": "#cdd451",
                  "babyblue": "#5ebcea", "orange": "#e58033", "yellow": "#eacf5e",
                  "raw": "#ece9d6"}
        self.project_colors_list = ["#58c1ae", "#0e73b9", "#cdd451", "#5ebcea", "#e58033","#eacf5e","#ece9d6"] 

    def initialize(self):
        raise NotImplementedError

    def evaluate(self, x):
        raise NotImplementedError

    def repair(self, x):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

        # @TODO ADDED BY LARISSA:
    def to_json(self, filename=None, path=None, json_=True, excel_=True):
        # Getting the attributes - It is a dict object:
        attributes_np = vars(self)
        
        attributes_dict = {}
        classname = self.__class__.__name__

        # Converting the value of the dict attributes from np.ndarray to list or handling dictionaries
        for key, value in attributes_np.items():
            if isinstance(value, np.ndarray):
                attributes_dict[key] = value.tolist()
            elif isinstance(value, dict):
                # If it's a dictionary, convert any numpy arrays inside it
                attributes_dict[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in value.items()}
            else:
                attributes_dict[key] = value
        
        # Define a custom serialization method
        def custom_serializer(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                return vars(obj)
            else:
                return str(obj)  # Fallback: Convert unknown types to strings
        
        # Converting to JSON with the custom serializer
        json_data = json.dumps(attributes_dict, indent=4, default=custom_serializer)

        if not filename:
            filename = classname
            
        if path:
            if not os.path.exists(path):
                os.makedirs(path)
            filename = os.path.join(path, filename)

        if json_:
            json_filename = filename + ".json"
            with open(json_filename, 'w') as json_file:
                json_file.write(json_data)

        if excel_:
            dfs_ = {}
            for key, value in attributes_dict.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], list):
                    dfs_[key] = pd.DataFrame(value)
                elif isinstance(value, dict):
                    dfs_[key] = pd.DataFrame([value])
                else:
                    dfs_[key] = pd.DataFrame({key: value if isinstance(value, list) else [value]})

            for df_ in dfs_.values():
                df_.columns = [f'Item {i}' for i in range(df_.shape[1])]

            excel_filename = filename + ".xlsx"
            
            with pd.ExcelWriter(excel_filename) as writer:
                for key_, _df in dfs_.items():
                    _df.to_excel(writer, sheet_name=key_, index=False)

    def exporting_results(self, path=None):

        for key, value in self.current_best.items():
            if isinstance(value, np.ndarray):
                self.current_best[key] = value.tolist()     

        #Cria um dicionário de DataFrames a partir do dicionário de arrays 2D
        dfs_ = {_key: pd.DataFrame(_value).T for _key, _value in self.current_best.items()}
        
        # Renomear as colunas em cada DataFrame
        for df_ in dfs_.values():
            df_.columns = [f'Item {i}' for i in range(df_.shape[1])]
        
        excel_filename = "current_best" + ".xlsx"
        if path:
            # Nome do arquivo Excel
            excel_filename = path + "current_best" + ".xlsx"
        
        # Salvar cada DataFrame em uma folha diferente
        with pd.ExcelWriter(excel_filename) as writer:
            for key_, _df in dfs_.items():  # Use dfs_ e não attributes_dict
                _df.to_excel(writer, sheet_name=key_, index=False)  # Salva cada DataFrame na folha correspondente          


    def export_profile_to_excel(self,
        genActPower,
        storDchActPower,
        v2gDchActPower,
        loadRedActPower,
        loadCutActPower,
        loadENS,
        pImp,
        genExcActPower,
        storChActPower,
        v2gChActPower,
        loadMax,
        name,
        path
        
    ):
          
        
        y1_prod = [sum(lista[j] for lista in genActPower) for j in range(len(genActPower[0]))]
        y2_prod = [sum(lista[j] for lista in storDchActPower) for j in range(len(storDchActPower[0]))]
        y3_prod = [sum(lista[j] for lista in v2gDchActPower) for j in range(len(v2gDchActPower[0]))]
        y4_prod = [sum(lista[j] for lista in loadRedActPower) for j in range(len(loadRedActPower[0]))]
        y5_prod = [sum(lista[j] for lista in loadCutActPower) for j in range(len(loadCutActPower[0]))]
        y6_prod = [sum(lista[j] for lista in loadENS) for j in range(len(loadENS[0]))]
        #y7_prod = result_pimp.values.reshape(model.t.last()-model.t.first()+1) 
        print("Opaaaa", len(loadMax))
        y1_cons = [sum(lista[j] for lista in loadMax) for j in range(len(loadMax[0]))]
        y2_cons = [sum(lista[j] for lista in genExcActPower) for j in range(len(genExcActPower[0]))]
        y3_cons = [sum(lista[j] for lista in storChActPower) for j in range(len(storChActPower[0]))]
        y4_cons = [sum(lista[j] for lista in v2gChActPower) for j in range(len(v2gChActPower[0]))]


        # Creating Production Dataframe
        df_production = pd.DataFrame({
            'Time [h]': list(range(1, len(y1_prod) + 1)),
            'Generators': y1_prod,
            'Storage': y2_prod,
            'V2G Discharge': y3_prod,
            'Load Reduction': y4_prod,
            'Load Cut': y5_prod,
            'Load ENS': y6_prod,
            'Imports': pImp
        })
        
        # Creating Consumption Dataframe
        df_consumption = pd.DataFrame({
            'Time [h]': list(range(1, len(y1_prod) + 1)),
            'Load': y1_cons,
            'Gen Excess': y2_cons,
            'Storage': y3_cons,
            'V2G Charge': y4_cons
        })

        full_path = os.path.join(path, name)

        if not os.path.exists(path):
            os.makedirs(path)
        
        with pd.ExcelWriter(full_path, engine='openpyxl') as writer:
            df_production.to_excel(writer, sheet_name='Production', index=False)
            df_consumption.to_excel(writer, sheet_name='Consumption', index=False)
        
        
    def plot (self,
        genActPower,
        storDchActPower,
        v2gDchActPower,
        loadRedActPower,
        loadCutActPower,
        loadENS,
        pImp,
        genExcActPower,
        storChActPower,
        v2gChActPower,
        loadMax,
        name,
        path, 
        graph_max=None,
        graph_step=None,
        save=False):
        
        y1_prod = np.array([sum(lista[j] for lista in genActPower) for j in range(len(genActPower[0]))])
        y2_prod = np.array([sum(lista[j] for lista in storDchActPower) for j in range(len(storDchActPower[0]))])
        y3_prod = np.array([sum(lista[j] for lista in v2gDchActPower) for j in range(len(v2gDchActPower[0]))])
        y4_prod = np.array([sum(lista[j] for lista in loadRedActPower) for j in range(len(loadRedActPower[0]))])
        y5_prod = np.array([sum(lista[j] for lista in loadCutActPower) for j in range(len(loadCutActPower[0]))])
        y6_prod = np.array([sum(lista[j] for lista in loadENS) for j in range(len(loadENS[0]))])
        
        y1_cons = [sum(lista[j] for lista in loadMax) for j in range(len(loadMax[0]))]
        y2_cons = np.array([sum(lista[j] for lista in genExcActPower) for j in range(len(genExcActPower[0]))])
        y3_cons = np.array([sum(lista[j] for lista in storChActPower) for j in range(len(storChActPower[0]))])
        y4_cons = np.array([sum(lista[j] for lista in v2gChActPower) for j in range(len(v2gChActPower[0]))])

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        # Plot Production
        axs[0].fill_between(list(range(1, len(y1_prod)+1)), np.zeros(len(y1_prod)), y1_prod, color=self.project_colors_list[0], label="Generator Power Production")
        axs[0].fill_between(list(range(1, len(y2_prod)+1)), y1_prod, y1_prod + y2_prod, color=self.project_colors_list[1], label="BESS Discharging Power")
        axs[0].fill_between(list(range(1, len(y3_prod)+1)), y1_prod + y2_prod, y1_prod + y2_prod + y3_prod, color=self.project_colors_list[2], label="EV Discharging Power")
        axs[0].fill_between(list(range(1, len(y4_prod)+1)), y1_prod + y2_prod + y3_prod, y1_prod + y2_prod + y3_prod + y4_prod, color=self.project_colors_list[3], label="Load Reduction")
        axs[0].fill_between(list(range(1, len(y5_prod)+1)), y1_prod + y2_prod + y3_prod + y4_prod, y1_prod + y2_prod + y3_prod + y4_prod + y5_prod, color=self.project_colors_list[4], label="Load Cut")
        axs[0].fill_between(list(range(1, len(y6_prod)+1)), y1_prod + y2_prod + y3_prod + y4_prod + y5_prod, y1_prod + y2_prod + y3_prod + y4_prod + y5_prod + y6_prod, color=self.project_colors_list[5], label="Load ENS")
        axs[0].fill_between(list(range(1, len(pImp)+1)), y1_prod + y2_prod + y3_prod + y4_prod + y5_prod + y6_prod, y1_prod + y2_prod + y3_prod + y4_prod + y5_prod + y6_prod + pImp, color=self.project_colors_list[6], label="Grid Import Power")
        axs[0].set_ylim(0, max(1.1*np.max(y1_prod + y2_prod + y3_prod + y4_prod + y5_prod + y6_prod + pImp), 1.1*np.max(y2_cons + y3_cons + y4_cons)))
        
        if graph_max and graph_step:
            axs[0].set_ylim(0, graph_max)
            axs[0].yaxis.set_major_locator(ticker.MultipleLocator(graph_step))
        
        axs[0].set_xlabel('Hour')
        axs[0].set_ylabel('Power [kW]')
        #axs[0].set_title('Production')
        axs[0].legend()

        # Plot Consumption

        axs[1].fill_between(list(range(1, len(y1_cons)+1)), np.zeros(len(y1_cons)), y1_cons, color=self.project_colors_list[0], label="Load Power Consumption")
        axs[1].fill_between(list(range(1, len(y2_cons)+1)), y1_cons, y1_cons + y2_cons, color=self.project_colors_list[1], label="Grid Export Power")
        axs[1].fill_between(list(range(1, len(y3_cons)+1)), y1_cons + y2_cons, y1_cons + y2_cons + y3_cons, color=self.project_colors_list[2], label="BESS Charging Power")
        axs[1].fill_between(list(range(1, len(y4_cons)+1)), y1_cons + y2_cons + y3_cons, y1_cons + y2_cons + y3_cons + y4_cons, color=self.project_colors_list[3], label="EV Charging Power")
        
        axs[1].set_ylim(0, max(1.1*np.max(y1_prod + y2_prod + y3_prod + y4_prod + y5_prod + y6_prod + pImp), 1.1*np.max(y1_cons + y2_cons + y3_cons + y4_cons)))
        if graph_max and graph_step:
            axs[1].set_ylim(0, graph_max)
            axs[1].yaxis.set_major_locator(ticker.MultipleLocator(graph_step))

            
        axs[1].set_xlabel('Hour')
        axs[1].set_ylabel('Power [kW]')
        #axs[1].set_title('Consumption')
        axs[1].legend()

        axs[0].set_xlim(1, len(genActPower[0]))
        axs[1].set_xlim(1, len(genActPower[0]))
        ticks = range(1, len(genActPower[0]) + 1)
        axs[0].set_xticks(ticks)
        axs[1].set_xticks(ticks)

        axs[0].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        axs[1].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

        plt.tight_layout()
        
        if save:
            full_path = os.path.join(path, name)

            if not os.path.exists(path):
                os.makedirs(path)

            # Save the plot to the specified path
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
        
        plt.show()