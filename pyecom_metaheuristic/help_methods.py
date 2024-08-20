import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

project_colors = {"green": "#58c1ae", "blue": "#0e73b9", "lightgreen": "#cdd451",
                  "babyblue": "#5ebcea", "orange": "#e58033", "yellow": "#eacf5e",
                  "raw": "#ece9d6"}
project_colors_list = ["#58c1ae", "#0e73b9", "#cdd451", "#5ebcea", "#e58033","#eacf5e","#ece9d6"] 

darkened_colors = ['#0c7462', '#00276c', '#808804', '#12709e', '#993400', '#9e8212', '#a09c8a']

def store_results(hour, results_dict,
                  result_pimp, result_pexp, result_genActPower, result_genExcActPower, 
                  result_genXo, result_loadRedActPower, result_loadCutActPower, 
                  result_loadENS, result_loadXo, result_storEnerState, 
                  result_storDchActPower, result_storChActPower, result_storDchXo, 
                  result_storChXo, 
                  #result_storScheduleChRelax, result_storScheduleDchRelax, 
                  result_v2gChActPower, result_v2gDchActPower, result_v2gEnerState, 
                  #result_v2gRelax, 
                  result_v2gDchXo, result_v2gChXo): 
                  #result_v2gScheduleChRelax, 
                  #result_v2gScheduleDchRelax, result_csActPower, result_csActPowerNet):
    

    results_dict[hour] = {
        'importsInit': np.array(result_pimp)[0],
        'exportsInit': np.array(result_pexp)[0],
        'genActPowerInit': np.array(result_genActPower)[:, 0],
        'genExcPowerInit': np.array(result_genExcActPower)[:, 0],
        'genXoInit': np.array(result_genXo)[:, 0],
        'loadRedActPowerInit': np.array(result_loadRedActPower)[:, 0],
        'loadCutActPowerInit': np.array(result_loadCutActPower)[:, 0],
        'loadENSInit': np.array(result_loadENS)[:, 0],
        'loadXoInit': np.array(result_loadXo)[:, 0],
        'storStateInit': np.array(result_storEnerState)[:, 0],
        'storDischargeInit': np.array(result_storDchActPower)[:, 0],
        'storChargeInit': np.array(result_storChActPower)[:, 0],
        'storRelaxInit': np.array(result_storDchXo)[:, 0],
        'storChXoInit': np.array(result_storChXo)[:, 0],
        'storDchXoInit': np.array(result_storDchXo)[:, 0],
        #'storScheduleChRelaxInit': np.array(result_storScheduleChRelax)[:, 0],
        #'storScheduleDchRelaxInit': np.array(result_storScheduleDchRelax)[:, 0],
        'v2gChargeInit': np.array(result_v2gChActPower)[:, 0],
        'v2gDischargeInit': np.array(result_v2gDchActPower)[:, 0],
        'v2gStateInit': np.array(result_v2gEnerState)[:, 0],
        #'v2gRelaxInit': np.array(result_v2gRelax)[:, 0],
        'v2gDchXoInit': np.array(result_v2gDchXo)[:, 0],
        'v2gChXoInit': np.array(result_v2gChXo)[:, 0],
        #'v2gScheduleChRelaxInit': np.array(result_v2gScheduleChRelax)[:, 0],
        #'v2gScheduleDchRelaxInit': np.array(result_v2gScheduleDchRelax)[:, 0],
        #'csChargeInit': np.array(result_csActPower)[:, 0],
        #'csNetChargeInit': np.array(result_csActPowerNet)[:, 0]
    }

    return results_dict



def plot_mixed_results(result_genActPower,
    result_storDchActPower,
    result_v2gDchActPower,
    result_loadRedActPower,
    result_loadCutActPower,
    result_loadENS,
    result_pimp,
    Data,
    _time_step,
    result_genExcActPower,
    result_storChActPower,
    result_v2gChActPower,
    save = False,
    path = None,
    name = None,
    graph_max = None,
    graph_step = None):

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # Production
    y1_prod = sum([result_genActPower[:, i].astype(float) for i in range(result_genActPower.shape[1])]) 
    y2_prod = sum([result_storDchActPower[:,i].astype(float) for i in range(result_storDchActPower.shape[1])]) 
    y3_prod = sum([result_v2gDchActPower[:, i].astype(float) for i in range(result_v2gDchActPower.shape[1])]) 
    y4_prod = sum([result_loadRedActPower[:, i].astype(float) for i in range(result_loadRedActPower.shape[1])]) 
    y5_prod = sum([result_loadCutActPower[:, i].astype(float) for i in range(result_loadCutActPower.shape[1])])
    y6_prod = sum([result_loadENS[:, i].astype(float) for i in range(result_loadENS.shape[1])]) 

    pImp = np.array(result_pimp, dtype=float)
    y7_prod = pImp.squeeze() 
    
    # Consumption
    #y1_cons = np.sum(Data.get_data().load['p_forecast'][:, 0:24*60//_time_step]*5, axis=0, dtype=np.float64)
    y1_cons = np.sum(Data.get_data().load['p_forecast'][:, 0:24*60//_time_step], axis=0, dtype=np.float64)
    y2_cons = sum([result_genExcActPower[:, i].astype(float) for i in range(result_genExcActPower.shape[1])])
    y3_cons = sum([result_storChActPower[:, i].astype(float) for i in range(result_storChActPower.shape[1])])
    y4_cons = sum([result_v2gChActPower[:, i].astype(float) for i in range(result_v2gChActPower.shape[1])]) 
    
    # Plot production

    axs[0].fill_between(list(range(1, len(y1_prod)+1)), np.zeros(len(y1_prod)), y1_prod, color=project_colors_list[0], edgecolor=darkened_colors[0], linestyle="dotted", label="Generator Power Production")
    axs[0].fill_between(list(range(1, len(y2_prod)+1)), y1_prod, y1_prod + y2_prod, color=project_colors_list[1], edgecolor=darkened_colors[1], linestyle="dotted",label="BESS Discharging Power")
    axs[0].fill_between(list(range(1, len(y3_prod)+1)), y1_prod + y2_prod, y1_prod + y2_prod + y3_prod, color=project_colors_list[2], edgecolor=darkened_colors[2], linestyle="dotted",label="EV Discharging Power")
    axs[0].fill_between(list(range(1, len(y4_prod)+1)), y1_prod + y2_prod + y3_prod, y1_prod + y2_prod + y3_prod + y4_prod, color=project_colors_list[3], edgecolor=darkened_colors[3], linestyle="dotted",label="Load Reduction")
    axs[0].fill_between(list(range(1, len(y5_prod)+1)), y1_prod + y2_prod + y3_prod + y4_prod, y1_prod + y2_prod + y3_prod + y4_prod + y5_prod, color=project_colors_list[4], edgecolor=darkened_colors[4], linestyle="dotted",label="Load Cut")
    axs[0].fill_between(list(range(1, len(y6_prod)+1)), y1_prod + y2_prod + y3_prod + y4_prod + y5_prod, y1_prod + y2_prod + y3_prod + y4_prod + y5_prod + y6_prod, color=project_colors_list[5], edgecolor=darkened_colors[5], linestyle="dotted",label="Load ENS")
    axs[0].fill_between(list(range(1, len(y7_prod)+1)), y1_prod + y2_prod + y3_prod + y4_prod + y5_prod + y6_prod, y1_prod + y2_prod + y3_prod + y4_prod + y5_prod + y6_prod + y7_prod, color=project_colors_list[6], edgecolor=darkened_colors[6], linestyle="dotted",label="Grid Import Power")

    axs[0].set_ylim(0, 1.1*max(np.max(y1_prod + y2_prod + y3_prod + y4_prod + y5_prod + y6_prod + y7_prod), np.max(y1_cons + y2_cons + y3_cons + y4_cons)))
    if graph_max and graph_step:
        axs[0].set_ylim(0, graph_max)
        axs[0].yaxis.set_major_locator(ticker.MultipleLocator(graph_step))
    

    axs[0].set_xlabel('Hour')
    axs[0].set_ylabel('Power [kW]')
    #axs[0].set_title('Production')
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    # Plot consumption
    axs[1].fill_between(list(range(1, len(y1_cons)+1)), np.zeros(len(y1_cons)), y1_cons, color=project_colors_list[0], edgecolor=darkened_colors[0], linestyle="dotted",label="Load Power Consumption")
    axs[1].fill_between(list(range(1, len(y2_cons)+1)), y1_cons, y1_cons + y2_cons, color=project_colors_list[1], edgecolor=darkened_colors[1], linestyle="dotted",label="Grid Export Power")
    axs[1].fill_between(list(range(1, len(y3_cons)+1)), y1_cons + y2_cons, y1_cons + y2_cons + y3_cons, color=project_colors_list[2], edgecolor=darkened_colors[2], linestyle="dotted",label="BESS Charging Power")
    axs[1].fill_between(list(range(1, len(y4_cons)+1)), y1_cons + y2_cons + y3_cons, y1_cons + y2_cons + y3_cons + y4_cons, color=project_colors_list[3], edgecolor=darkened_colors[3], linestyle="dotted",label="EV Charging Power")



    axs[1].set_ylim(0, 1.1*max(np.max(y1_prod + y2_prod + y3_prod + y4_prod + y5_prod + y6_prod + y7_prod), np.max(y1_cons + y2_cons + y3_cons + y4_cons)))    
    if graph_max and graph_step:
        axs[1].set_ylim(0, graph_max)
        axs[1].yaxis.set_major_locator(ticker.MultipleLocator(graph_step))

    axs[1].set_ylim(0, graph_max)
    axs[1].yaxis.set_major_locator(ticker.MultipleLocator(graph_step))
    axs[1].set_xlabel('Hour')
    axs[1].set_ylabel('Power [kW]')
    #axs[1].set_title('Consumption')
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)

    axs[0].set_xlim(1, 24*60//_time_step)
    axs[1].set_xlim(1, 24*60//_time_step)

    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    plt.tight_layout()
    ticks = range(1, 24*60//_time_step + 1)
    axs[0].set_xticks(ticks)
    axs[1].set_xticks(ticks)
    
    full_path = os.path.join(path, name)

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(path):
        os.makedirs(path)

    # Save the plot to the specified path
    plt.savefig(full_path, dpi=300, bbox_inches='tight')

    plt.show()

def export_mixed_results(result_genActPower,
    result_storDchActPower,
    result_v2gDchActPower,
    result_loadRedActPower,
    result_loadCutActPower,
    result_loadENS,
    result_pimp,
    Data,
    _time_step,
    result_genExcActPower,
    result_storChActPower,
    result_v2gChActPower,
    path = None,
    name = None):
    # Production
    y1_prod = sum([result_genActPower[:, i].astype(float) for i in range(result_genActPower.shape[1])]) 
    y2_prod = sum([result_storDchActPower[:,i] for i in range(result_storDchActPower.shape[1])]) 
    y3_prod = sum([result_v2gDchActPower[:, i] for i in range(result_v2gDchActPower.shape[1])]) 
    y4_prod = sum([result_loadRedActPower[:, i] for i in range(result_loadRedActPower.shape[1])]) 
    y5_prod = sum([result_loadCutActPower[:, i] for i in range(result_loadCutActPower.shape[1])])
    y6_prod = sum([result_loadENS[:, i] for i in range(result_loadENS.shape[1])]) 

    y7_prod = result_pimp.squeeze() 

    # Consumption
    #y1_cons = np.sum(Data.get_data().load['p_forecast'][:, 0:24*60//_time_step]*5, axis=0, dtype=np.float64)
    y1_cons = np.sum(Data.get_data().load['p_forecast'][:, 0:24*60//_time_step], axis=0, dtype=np.float64)
    y2_cons = sum([result_genExcActPower[:, i] for i in range(result_genExcActPower.shape[1])])
    y3_cons = sum([result_storChActPower[:, i] for i in range(result_storChActPower.shape[1])])
    y4_cons = sum([result_v2gChActPower[:, i] for i in range(result_v2gChActPower.shape[1])]) 

    # Creating Production Dataframe
    df_production = pd.DataFrame({
        'Time [h]': list(range(1, len(y1_prod) + 1)),
        'Generators': y1_prod,
        'Storage': y2_prod,
        'V2G Discharge': y3_prod,
        'Load Reduction': y4_prod,
        'Load Cut': y5_prod,
        'Load ENS': y6_prod,
        'Imports': y7_prod
    })
    
    # Creating Consumption Dataframe
    df_consumption = pd.DataFrame({
        'Time [h]': list(range(1, len(y1_cons) + 1)),
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