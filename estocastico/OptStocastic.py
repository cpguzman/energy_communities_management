import pyomo.environ as pyo
import pandas as pd
import dotenv

conf = dotenv.dotenv_values()
cplex_path = conf["PATH_TO_SOLVER"]

model = pyo.ConcreteModel("Energy Community")

#Data
General_Information_Data_excel =  pd.ExcelFile("./Dados/General_Information Data.xlsx")
df_P_Max_Imp = General_Information_Data_excel.parse("1 P Max_Imp (kW)",index_col=0)
dict_P_Max_Imp = df_P_Max_Imp.to_dict()
df_P_Max_Exp = General_Information_Data_excel.parse("2 P Max_Exp (kW)",index_col=0)
dict_P_Max_Exp = df_P_Max_Exp.to_dict()
Load_Data_excel =  pd.ExcelFile("./Dados/Load_Data.xlsx")
df_Pload_Forecast = Load_Data_excel.parse("1 P Forecast (kW)",index_col=0)
Pload_Forecast_dict = df_Pload_Forecast.to_dict()
Generators_Data_excel =  pd.ExcelFile("./Dados/Generators_Data.xlsx")
df_GenCharacteritics = Generators_Data_excel.parse("9 Characteritics D.",index_col=0)
GenCharacteritics_dict = df_GenCharacteritics.to_dict()
df_Pgen_Forecast = Generators_Data_excel.parse("1 P Forecast (kW)",index_col=0)
Pgen_Forecast_dict = df_Pgen_Forecast.to_dict()
df_GenCost = Generators_Data_excel.parse("3 Cost Parameter B (m.u.)",index_col=0)
Pgen_Cost_dict = df_GenCost.to_dict()
df_P_Imp_Price = General_Information_Data_excel.parse("3 Energy_Buy_Price (m.u.)",index_col=0)
dict_C_PImp_Price = df_P_Imp_Price.to_dict()
df_P_Exp_Price = General_Information_Data_excel.parse("4 Energy_Sell_Price (m.u.)",index_col=0)
dict_C_PExp_Price = df_P_Exp_Price.to_dict()
# Defining Sets
nW = 3
model.SetSceneries = pyo.Set(initialize=list(range(nW)), doc='Set of Scenarios')
model.SetTimeIntervals = pyo.Set(initialize=df_Pload_Forecast.columns, doc='Set of time intervals')
model.LoadSet = pyo.Set(initialize=df_Pload_Forecast.index, doc='Set of Loads')
model.GenSet = pyo.Set(initialize=df_Pgen_Forecast.index, doc='Set of Generatos')
# Defining the parameter
@model.Param(model.GenSet)
def Param_P_MaxGen_g(model, gen):
  return GenCharacteritics_dict['6 P Max. (kW)'][gen]

@model.Param(model.GenSet, model.SetTimeIntervals)
def ParamC_UpGenCost_g_t(model, generator, time):
  return Pgen_Cost_dict[time][generator]

@model.Param(model.GenSet, model.SetTimeIntervals)
def ParamC_DownGenCost_g_t(model, generator, time):
  return Pgen_Cost_dict[time][generator]

@model.Param(model.SetTimeIntervals)
def Param_P_MaxExp_t(model, time):
  return dict_P_Max_Exp[time]['PCC_0']

@model.Param(model.LoadSet, model.SetTimeIntervals, within=pyo.NonNegativeReals)
def Param_Pld_l_t(model, load, time):
  return Pload_Forecast_dict[time][load]

@model.Param(model.SetTimeIntervals)
def Param_P_MaxImp_t(model, time):
  return dict_P_Max_Imp[time]['PCC_0']

@model.Param(model.SetTimeIntervals)
def Param_C_ImpPrice_t(model, time):
  return dict_C_PImp_Price[time]['PCC_0']

@model.Param(model.SetTimeIntervals)
def Param_c_UpImp_t(model, time):
  return dict_C_PImp_Price[time]['PCC_0']

@model.Param(model.SetTimeIntervals)
def Param_c_DownImp_t(model, time):
  return dict_C_PImp_Price[time]['PCC_0']

@model.Param(model.SetTimeIntervals)
def Param_C_UpImpPrice_t(model, time):
  return dict_C_PImp_Price[time]['PCC_0']

@model.Param(model.SetTimeIntervals)
def Param_C_DownImpPrice_t(model, time):
  return dict_C_PImp_Price[time]['PCC_0']

@model.Param(model.SetTimeIntervals)
def Param_C_ExpPrice_t(model, time):
  return dict_C_PExp_Price[time]['PCC_0']

@model.Param(model.GenSet, model.SetTimeIntervals)
def Param_c_Gent_g_t(model, generator, time):
  return Pgen_Cost_dict[time][generator]

@model.Param(model.GenSet, model.SetTimeIntervals)
def Param_c_Gen_Up_g_t(model, generator, time):
  return Pgen_Cost_dict[time][generator]

@model.Param(model.GenSet, model.SetTimeIntervals)
def Param_c_Gen_Down_g_t(model, generator, time):
  return Pgen_Cost_dict[time][generator]

# Defining Decision Variables
model.VarP_OpGen_g_t = pyo.Var(model.GenSet, model.SetTimeIntervals, domain = pyo.NonNegativeReals, initialize=0)

def UpGen_bounds(model, gen, time):
    lower_bound = 0
    upper_bound = model.Param_P_MaxGen_g[gen]
    return (lower_bound, upper_bound)

model.VarR_UpGen_g_t = pyo.Var(model.GenSet, model.SetTimeIntervals, bounds = UpGen_bounds, domain = pyo.NonNegativeReals, initialize=0)

def DownGen_bounds(model, gen, time):
    lower_bound = 0
    upper_bound = model.Param_P_MaxGen_g[gen]
    return (lower_bound, upper_bound)

model.VarR_DownGen_g_t = pyo.Var(model.GenSet, model.SetTimeIntervals, bounds = DownGen_bounds, domain = pyo.NonNegativeReals, initialize=0)

def R_Up_Import_bounds(model, time):
  lower_bound = 0
  upper_bound = model.Param_P_MaxImp_t[time]
  return (lower_bound, upper_bound)

model.VarR_UpImp_t = pyo.Var(model.SetTimeIntervals, domain = pyo.NonNegativeReals, bounds = R_Up_Import_bounds, initialize=0)

def R_Down_Import_bounds(model, time):
  lower_bound = 0
  upper_bound = model.Param_P_MaxImp_t[time]
  return (lower_bound, upper_bound)

model.VarR_DownImp_t = pyo.Var(model.SetTimeIntervals, domain = pyo.NonNegativeReals, bounds = R_Down_Import_bounds, initialize=0)

model.Var_r_UpGen_g_t_w = pyo.Var(model.GenSet, model.SetTimeIntervals, model.SetSceneries, domain = pyo.NonNegativeReals, initialize=0)

model.Var_r_DownGen_g_t_w = pyo.Var(model.GenSet, model.SetTimeIntervals, model.SetSceneries, domain = pyo.NonNegativeReals, initialize=0)

model.VarP_Op_imp_t = pyo.Var(model.SetTimeIntervals, domain = pyo.NonNegativeReals, initialize=0)

model.Var_r_Up_imp_t_w = pyo.Var(model.SetTimeIntervals, model.SetSceneries, domain = pyo.NonNegativeReals, initialize=0)

model.Var_r_Down_imp_t_w = pyo.Var(model.SetTimeIntervals, model.SetSceneries, domain = pyo.NonNegativeReals, initialize=0)

model.VarP_Op_Exp_t = pyo.Var(model.SetTimeIntervals, domain = pyo.NonNegativeReals, initialize=0)

model.VarP_Op_Grid_t = pyo.Var(model.SetTimeIntervals, domain = pyo.Reals, initialize=0)

model.XopGrid_t =  pyo.Var(model.SetTimeIntervals, within = pyo.Binary, initialize = 0)

def rule_power_reserve_Up(model, gen, time, w):
  return model.Var_r_UpGen_g_t_w[gen, time, w] <= model.VarR_UpGen_g_t[gen, time]
model.power_reserve_up_constraint = pyo.Constraint(model.GenSet, model.SetTimeIntervals, model.SetSceneries, rule=rule_power_reserve_Up)

def rule_power_reserve_Down(model, gen, time, w):
  return model.Var_r_DownGen_g_t_w[gen, time, w] <= model.VarR_DownGen_g_t[gen, time]
model.power_reserve_Down_constraint = pyo.Constraint(model.GenSet, model.SetTimeIntervals, model.SetSceneries, rule=rule_power_reserve_Down)

def rule_power_reserve_Up_import(model,time, w):
  return model.Var_r_Up_imp_t_w[time, w] <= model.VarR_UpImp_t[time]
model.power_reserve_Up_import_constraint = pyo.Constraint(model.SetTimeIntervals, model.SetSceneries, rule=rule_power_reserve_Up_import)

def rule_power_reserve_Down_import(model,time, w):
  return model.Var_r_Down_imp_t_w[time, w] <= model.VarR_DownImp_t[time]
model.power_reserve_Down_import_constraint = pyo.Constraint(model.SetTimeIntervals, model.SetSceneries, rule=rule_power_reserve_Down_import)

def rule_gen_power_max(model, gen, time, w):
  return model.VarP_OpGen_g_t[gen, time] + model.Var_r_UpGen_g_t_w[gen, time, w] <= model.Param_P_MaxGen_g[gen]
model.gen_power_max_constraint = pyo.Constraint(model.GenSet, model.SetTimeIntervals, model.SetSceneries, rule=rule_gen_power_max)

def rule_gen_power_min(model, gen, time, w):
  return model.VarP_OpGen_g_t[gen, time] - model.Var_r_DownGen_g_t_w[gen, time, w] >= 0
model.gen_power_min_constraint = pyo.Constraint(model.GenSet, model.SetTimeIntervals, model.SetSceneries, rule=rule_gen_power_min)

def rule_power_imp(model, time, w):
  return model.VarP_Op_imp_t[time] - model.Var_r_Up_imp_t_w[time, w] >= 0
model.power_imp_constraint = pyo.Constraint(model.SetTimeIntervals, model.SetSceneries, rule=rule_power_imp)

def rule_power_reserve_Down_import_Op(model,time, w):
  return model.VarP_Op_imp_t[time] - model.Var_r_Down_imp_t_w[time, w] >= 0
model.power_reserve_Down_import_Op_import_constraint = pyo.Constraint(model.SetTimeIntervals, model.SetSceneries, rule=rule_power_reserve_Down_import_Op)

def rule_power_export(model, time):
  return model.VarP_Op_Exp_t[time] <= model.Param_P_MaxExp_t[time]
model.power_export_constraint = pyo.Constraint(model.SetTimeIntervals, rule=rule_power_export)

def rule_imported_power(model, time):
  return model.VarP_Op_imp_t[time] == (model.VarP_Op_Grid_t[time] * model.XopGrid_t[time])
model.imported_power_constraint = pyo.Constraint(model.SetTimeIntervals, rule=rule_imported_power)

def rule_exported_power(model, time):
  return model.VarP_Op_Exp_t[time] == -1* (model.VarP_Op_Grid_t[time] * (1-model.XopGrid_t[time]))
model.exported_power_constraint = pyo.Constraint(model.SetTimeIntervals, rule=rule_exported_power)

def rule_balance_power(model, time, w):
  total_load = sum(model.Param_Pld_l_t[load, time] for load in model.LoadSet)
  GenPower = sum(model.VarP_OpGen_g_t[gen, time] + model.Var_r_UpGen_g_t_w[gen, time, w] for gen in model.GenSet) - model.Var_r_Down_imp_t_w[time, w]
  return model.VarP_Op_Grid_t[time] + model.Var_r_Up_imp_t_w[time, w] - model.Var_r_Down_imp_t_w[time, w] == total_load - GenPower

model.balance_power_constraint = pyo.Constraint(model.SetTimeIntervals, model.SetSceneries, rule=rule_balance_power)

# Objective Function Minimization of Cost
def objective_rule(model):
  CostoRUpGenPower = sum(model.VarR_UpGen_g_t[gen, time] * model.ParamC_UpGenCost_g_t[gen, time] for gen in model.GenSet for time in model.SetTimeIntervals)
  CostoRDownGenPower = sum(model.VarR_DownGen_g_t[gen, time] * model.ParamC_DownGenCost_g_t[gen, time] for gen in model.GenSet for time in model.SetTimeIntervals)
  CostoReservaPowerImport = sum(model.VarR_UpImp_t[time] * model.Param_C_UpImpPrice_t[time] + model.VarR_DownImp_t[time] * model.Param_C_DownImpPrice_t[time] for time in model.SetTimeIntervals)
  FDA = CostoRUpGenPower + CostoRDownGenPower
  CostoGenPower = 0
  CostoPowerImport = 0
  CostoPowerExport = 0
  pi_w = 1/nW
  for w in model.SetSceneries:
    CostoGenPower = CostoGenPower + pi_w * sum(model.Var_r_UpGen_g_t_w[gen, time, w] * model.Param_c_Gen_Up_g_t[gen, time]  + model.Var_r_DownGen_g_t_w[gen, time, w] * model.Param_c_Gen_Down_g_t[gen, time] + model.VarP_OpGen_g_t[gen, time] * model.Param_c_Gent_g_t[gen, time] for gen in model.GenSet for time in model.SetTimeIntervals)
    CostoPowerImport = CostoPowerImport + pi_w *sum(model.Var_r_Up_imp_t_w[time, w] * model.Param_c_UpImp_t[time] + model.Var_r_Down_imp_t_w[time, w] * model.Param_c_DownImp_t[time] + model.VarP_Op_imp_t[time] * model.Param_C_ImpPrice_t[time] for time in model.SetTimeIntervals)
    CostoPowerExport = CostoPowerExport + pi_w *sum(model.VarP_Op_Exp_t[time] * model.Param_C_ExpPrice_t[time] for time in model.SetTimeIntervals)
  FRT = CostoGenPower + CostoPowerImport - CostoPowerExport
  return FDA + FRT

model.Objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
solver = pyo.SolverFactory('cplex', executable=cplex_path)
#solver = pyo.SolverFactory('ipopt')
#solver.set_executable(cplex_path, validate=False)
results = solver.solve(model, tee=True)
print(results)
# Create a list to store the data
data = dict()
columnas = list()

variables = [model.XopGrid_t]
for variable in variables:
  columnas.append(variable.name)
  data_variable = list()
  for index in variable:
    value = pyo.value(variable[index])
    data_variable.append(value)
  data[variable.name]=data_variable

# Convert the list of dictionaries into a DataFrame
df = pd.DataFrame(data, columns=columnas)
print(df)