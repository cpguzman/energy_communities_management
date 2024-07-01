#**************************************************************
#                        Import packages
#**************************************************************
import pandas as pd
import pyomo.environ as pyo
#import numpy as np
#**************************************************************
#                        Data definition
#**************************************************************
General_Information_Data_excel = pd.ExcelFile("Data/General_Information Data.xlsx")
df_P_Max_Imp = General_Information_Data_excel.parse("1 P Max_Imp (kW)",index_col=0)
df_P_Max_Exp = General_Information_Data_excel.parse("2 P Max_Exp (kW)",index_col=0)
df_P_Imp_Price = General_Information_Data_excel.parse("3 Energy_Buy_Price (m.u.)",index_col=0)
df_P_Exp_Price = General_Information_Data_excel.parse("4 Energy_Sell_Price (m.u.)",index_col=0)
dict_P_Max_Imp = df_P_Max_Imp.to_dict()
dict_P_Max_Exp = df_P_Max_Exp.to_dict()
dict_C_PImp_Price = df_P_Imp_Price.to_dict()
dict_C_PExp_Price = df_P_Exp_Price.to_dict()

Load_Data_excel = pd.ExcelFile("Data/Load_Data.xlsx")
df_Pload_Forecast = Load_Data_excel.parse("1 P Forecast (kW)",index_col=0)
dict_Pload_Forecast = df_Pload_Forecast.to_dict()

V2G_Data_excel = pd.ExcelFile("Data/V2G_Data.xlsx")
df_V2G_ConexionStatus = V2G_Data_excel.parse("1 ConexionStatusV2G",index_col=0)
df_V2G_EinicialArriving = V2G_Data_excel.parse("2 Energy Inicial Arriving",index_col=0)
df_V2G_ExitRequired = V2G_Data_excel.parse("3 Energy Exit Required ",index_col=0)
df_V2G_ChargePrice = V2G_Data_excel.parse("6 Charge Price",index_col=0)
df_V2G_DischargePrice = V2G_Data_excel.parse("7 Disharge Price",index_col=0)
df_V2G_Characteritics = V2G_Data_excel.parse("8 Characteritics D.",index_col=0)
dict_V2G_ConexionStatus = df_V2G_ConexionStatus.to_dict()
dict_V2G_EinicialArriving = df_V2G_EinicialArriving.to_dict()
dict_V2G_ExitRequired = df_V2G_ExitRequired.to_dict()
dict_V2G_ChargePrice = df_V2G_ChargePrice.to_dict()
dict_V2G_DischargePrice = df_V2G_DischargePrice.to_dict()
dict_V2G_Characteritics = df_V2G_Characteritics.to_dict()

Storage_Data_excel = pd.ExcelFile("Data/Storage_Data.xlsx")
df_PstorageChargeLimit = Storage_Data_excel.parse("1 P Charge Limit (kW)",index_col=0)
df_PstorageDischargeLimit = Storage_Data_excel.parse("2 P Discharge Limit (kW)",index_col=0)
df_PstorageChargePrice = Storage_Data_excel.parse("3 Charge price (m.u)",index_col=0)
df_PstorageDischargePrice = Storage_Data_excel.parse("4 Discharge price (m.u.)",index_col=0)
df_StorageCharacteritics = Storage_Data_excel.parse("5 Characteritics D.",index_col=0)
dict_PstorageChargeLimit = df_PstorageChargeLimit.to_dict()
dict_PstorageDischargeLimit = df_PstorageDischargeLimit.to_dict()
dict_PstorageChargePrice = df_PstorageChargePrice.to_dict()
dict_PstorageDischargePrice = df_PstorageDischargePrice.to_dict()
dict_StorageCharacteritics = df_StorageCharacteritics.to_dict()

Generators_Data_excel = pd.ExcelFile("Data/Generators_Data.xlsx")
df_Pgen_Forecast = Generators_Data_excel.parse("1 P Forecast (kW)",index_col=0)
df_GenCost = Generators_Data_excel.parse("3 Cost Parameter B (m.u.)",index_col=0)
df_CostGenerationExcess = Generators_Data_excel.parse("5 Cost NDE (m.u.)",index_col=0)
df_GenCharacteritics = Generators_Data_excel.parse("9 Characteritics D.",index_col=0)
dict_Pgen_Forecast = df_Pgen_Forecast.to_dict()
dict_Pgen_Cost = df_GenCost.to_dict()
dict_CostGenerationExcess = df_CostGenerationExcess.to_dict()
dict_GenCharacteritics = df_GenCharacteritics.to_dict()

ChargeStation_Data_excel = pd.ExcelFile("Data/ChargeStation_Data.xlsx")
df_PcsChargeLimit = ChargeStation_Data_excel.parse("1 P Charge Limit (kW)",index_col=0)
df_CS_Characteritics = ChargeStation_Data_excel.parse("3 Characteritics D.",index_col=0)
dict_PcsChargeLimit = df_PcsChargeLimit.to_dict()
dict_CS_Characteritics = df_CS_Characteritics.to_dict()
#**************************************************************
#                        Sets definition
#**************************************************************
model = pyo.ConcreteModel()
model.SetTimeIntervals = pyo.Set(initialize=df_Pload_Forecast.columns, doc='Set of time intervals')
model.LoadSet = pyo.Set(initialize=df_Pload_Forecast.index, doc='Set of Loads')
model.GenSet = pyo.Set(initialize=df_Pgen_Forecast.index, doc='Set of Generatos')
model.BatterySet = pyo.Set(initialize=df_PstorageChargeLimit.index, doc='Set of Energy Storage System')
model.V2G_Set = pyo.Set(initialize=df_V2G_Characteritics.index, doc='Set of Electric Vehicles')
model.ChargeStation_Set = pyo.Set(initialize=df_CS_Characteritics.index, doc='Set of Charge Station')
#**************************************************************
#                        Parameters definition
#**************************************************************
@model.Param(model.LoadSet, model.SetTimeIntervals)
def ParamPloadForecast_l_t(model, load, time):
  return dict_Pload_Forecast[time][load]

@model.Param(model.BatterySet, model.SetTimeIntervals)
def ParamPstorageChargeLimit_b_t(model, storage, time):
  return dict_PstorageChargeLimit[time][storage]

@model.Param(model.BatterySet, model.SetTimeIntervals)
def ParamPstorageDischargeLimit_b_t(model, storage, time):
  return dict_PstorageDischargeLimit[time][storage]

@model.Param(model.BatterySet)
def EstorageMax_b(model, storage):
  return dict_StorageCharacteritics["6 Energy Capacity (kVAh)"][storage]

@model.Param(model.V2G_Set)
def Pv2gDischargeMax_v(model, v2g):
  return dict_V2G_Characteritics["7 P Discharge Max (kW)"][v2g]

@model.Param(model.V2G_Set)
def Pv2gChargeMax_v(model, v2g):
  return dict_V2G_Characteritics["6 P Charge Max (kW)"][v2g]

@model.Param(model.V2G_Set, model.SetTimeIntervals)
def XconexionStatus_v_t(model, v2g, time):
  return dict_V2G_ConexionStatus[time][v2g]

@model.Param(model.V2G_Set)
def Ev2gMax_v(model, v2g):
  return dict_V2G_Characteritics["5 E Capacity Max (kWh)"][v2g]

@model.Param(model.V2G_Set, model.SetTimeIntervals)
def Ev2gInitial_v_t(model, v2g, time):
  return dict_V2G_EinicialArriving[time][v2g]

@model.Param(model.V2G_Set, model.SetTimeIntervals)
def ParamEv2gExitRequired_v_t(model, v2g, time):
  return dict_V2G_ExitRequired[time][v2g]

@model.Param(model.GenSet, model.SetTimeIntervals)
def ParamPGenForecast_g_t(model, generator, time):
  return dict_Pgen_Forecast[time][generator]

@model.Param(model.GenSet, model.SetTimeIntervals)
def ParamPGenCost_g_t(model, generator, time):
  return dict_Pgen_Cost[time][generator]








print("fin")




#print("GENERAL ", ChargeStation_Data_excel.sheet_names)
#**************************************Data********************
