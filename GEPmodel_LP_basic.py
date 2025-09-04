# -*- coding: utf-8 -*-
"""
Unit Conversions :
  Input File        Python Code Model  
  -------------     --------------------------------------------------
  Dollars       ->  million dollars        million $ = 1e-6 $
  MW            ->  GW                     GW = 1e-3 MW
  $/MW          ->  million $/GW           million $ / GW = 1e-3 $/MW
  kgCO2/MW      ->  tonCO2/GW              tonCO2/GW = kgCO2/MW
  kgCO2         ->  tonCO2                 tonCO2 = 1e-3 kgCO2
  kgNOX/MW      ->  tonNOX/GW              tonNOX/GW = kgNOX/MW
  kgNOX         ->  tonNOX                 tonNOX = 1e-3 kgNOX

t0 = 2025

"""

#%% IMPORT LIBRARIES
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
plt.rcParams.update({'font.size': 14})
import os
import psutil

#%% Functions

# get_results FUNCTION
def get_results(get_v=True, get_y=True, get_u=True):
    """
    get the results (variable values) from the optimum solution

    Parameters:
    get_v: Boolean (if we want to get generation amounts) default: True
    get_y: Boolean (if we want to get transmission amounts) default: False
    get_u: Boolean (if we want to get unmet demand amounts) default: True
    No parameter is passed for the model object we want the results for, since
    this function is called right after solving the model once we have solution

    Returns:
    df_s : dataframe for s variables (includes only t,b,k s.t. s[t,b,k] > 0) 
           dataframe columns --> t,b,k: indices, s: new investment value in GW
    df_ce: dataframe for cost and emission variables in each t (includes all t) 
           dataframe columns --> t: index
           all cost in million USD
           IC  : investment cost in t
           MC  : maintanence cost in t	
           PC  : investment cost in t
           TC  : investment cost in t
           CS  : investment cost in t
           NetCost  : investment cost in t
           NPV  : investment cost in t
           CO2 emissions in tonCO2 and NOX emissions in tonNOX
           GCE : CO2 emissions from electricity generation in t
           NetCO2 : Net CO2 emissions in t (GCE - VCES)
           VCES: CO2 saving due to EVs in t
           GNE : NOX emissions from electricity generation in t
           VNES: NOX saving due to EVs in t
           NetNOx : Net NOX emissions in t (GNE - VNES)
    df_v : dataframe for v variables (includes only t,g,h,k s.t. v[t,g,h,:,:,k] > 0) 
           dataframe columns --> t,g,h,k: indices, v: generation value in GWh
    df_y : dataframe for y variables (includes only t,g,h,b1,b2 s.t. y[t,g,h,:,b1,b2] > 0) 
           dataframe columns --> t,g,h,b1,b2: indices, y: transmission value in GWh
    df_u : dataframe for u variables (includes only t,g,h,b s.t. v[t,g,h,:,b] > 0) 
           dataframe columns --> t,g,h,b: indices, v: unmet demand value in GWh
    df_slack_NC : dataframe for slack values of Regional New Capacity Constraints
                  dataframe columns --> t_strt, t_end, b, NC, slack, dual, slack_to_other_regin", dual_to_other_region, slack_to_other_twindow, dual_to_other_twindow
    df_NC_tr : dataframe for transferred Regional New Capacity values
               dataframe columns --> tstrt, tend, from_b, "to_b, NC_tr 
    """
    
    # investment decisions
    s_values = s.X
    df_s = pd.DataFrame(columns=["t", "b", "k", "s"])
    for t in range(len(tSet)):
        for b in range(len(B)):
            for k in range(len(subK)):
                if s_values[t,b,k] > 0:
                    df_s.loc[len(df_s)] = [ int(t), B[b], subK[k], s_values[t,b,k] ]
                
    # generation
    df_v = pd.DataFrame(columns=["t", "g", "h", "k", "v"])
    if get_v == True:
        v_values = v.X
        weighted_v = (weight_tj[:,None,None,:,None] * v_values.sum(axis=4)).sum(axis=3)
        for t in range(len(tSet)):
            for g in range(len(subG)):
                for h in range(len(H)):
                    for k in range(len(K)):
                        df_v.loc[len(df_v)] = [ int(t), subG[g], H[h], K[k], weight_g[g] * weighted_v[t,g,h,k] ]
    
    # transmission
    df_y = pd.DataFrame(columns=["t", "g", "h", "b0", "b1", "y"])
    if get_y == True:
        y_values = y.X
        weighted_y = (weight_tj[:,None,None,:,None,None] * y_values).sum(axis=3)
        for t in range(len(tSet)):
            for g in range(len(subG)):
                for h in range(len(H)):
                    for j in range(len(J)):
                        for b in B:
                            for bb in A[b]:
                                if bb != b:
                                    df_y.loc[len(df_y)] = [ int(t), subG[g], H[h], b, bb, weight_g[g] * weighted_y[t,g,h,b,bb] ]
    
    # unmet demand
    df_u = pd.DataFrame(columns=["t", "g", "h", "u"])
    if get_u == True:
        u_values = u.X
        weighted_u = (weight_tj[:,None,None,:] * u_values.sum(axis=4)).sum(axis=3)
        for t in range(len(tSet)):
            for g in range(len(subG)):
                for h in range(len(H)):
                    df_u.loc[len(df_u)] = [ int(t), subG[g], H[h], weight_g[g] * weighted_u[t,g,h] ]

    # cost and emissions
    df_ce = pd.DataFrame({
        "t": tSet,
        "IC": TIC.X,
        "MC": TMC.X,
        "PC": TPC.X,
        "UC": TUC.X,
        "TC": TFTC.X + TVTC.X,
        "CC": TCC.X,
        "CS": CS.X,
        "NetCost": TIC.X + TMC.X + TPC.X + TUC.X + TFTC.X + TVTC.X + TCC.X - CS.X,
        "NPV": [((1 + r) ** (-t)) * (TIC.X[t] + TMC.X[t] + TPC.X[t] + TUC.X[t] + TFTC.X[t] + TVTC.X[t] + TCC.X[t] - CS.X[t]) for t in tSet],
        "GCE": GCE.X,
        "VCES": VCES.X,
        "NetCO2": GCE.X - VCES.X,
        "GNE": GNE.X,
        "VNES": VNES.X,
        "NetNOx": GNE.X - VNES.X
    })
    
    df_slack_NC = pd.DataFrame(columns=["t_strt", "t_end", "b", "NC", "slack", "dual", "slack_to_other_region", "dual_to_other_region", "slack_to_other_twindow", "dual_to_other_twindow"])
    for (t_strt, t_end, b) in NC_:
        if t_end in tSet:
            consName1 = f'new_connectable_cap_{t_strt}_{t_end}_{B.index(b)}'
            constraint1 = gep.getConstrByName(consName1)
            consName2 = f"trans_connectable_cap_otherregions_{t_strt}_{t_end}_{B.index(b)}"
            constraint2 = gep.getConstrByName(consName2)
            consName3 = f"trans_connectable_cap_othertwidows_{t_strt}_{t_end}_{B.index(b)}"
            constraint3 = gep.getConstrByName(consName3)
            if constraint1:
                slack_val1 = constraint1.Slack
                dual_val1 = constraint1.Pi
            if constraint2:
                slack_val2 = constraint2.Slack
                dual_val2 = constraint2.Pi
            if constraint3:
                slack_val3 = constraint3.Slack
                dual_val3 = constraint3.Pi
            df_slack_NC.loc[len(df_slack_NC)] = [ int(t_strt), int(t_end), b, NC_[(t_strt, t_end, b)], slack_val1, dual_val1, slack_val2, dual_val2, slack_val3, dual_val3 ]

    NC_tr_values = NC_reg_tr.X
    df_NC_tr = pd.DataFrame(columns=["t", "tt", "b", "bb", "NC_tr"])
    for t in tSet:
        for tt in tSet:
            for b in B:
                for bb in B:
                    if NC_tr_values[t, tt, B.index(b), B.index(bb)] > 0:
                        df_NC_tr.loc[len(df_NC_tr)] = [ t, tt, b, bb, NC_tr_values[t, tt, B.index(b), B.index(bb)] ]
                        
    return df_s, df_ce, df_v, df_y, df_u, df_slack_NC, df_NC_tr

# save_results FUNCTION
def save_results(t, w, df_s, df_ce, df_v, df_y, df_u, df_slack_NC, df_NC_tr):   
    """
    save the results of the current solution to an excel output file
    and append model weight combination and obj function values for Cost, CO2 and NOX to results dictionary 
    
    Parameters:
    t : max years (length of the planning period in the model)
    n : max penetration level in the model
    w : weight array for obj func criteria        
    df_s, df_x, df_ce, df_v, df_y, df_u, df_slack_NC, df_NC_tr dataframes returned from get_results function
    
    Returns:
    None
    
    Creates an output file (.xlsx) that includes dataframes in each sheet
    adds result to the results dictionary (already defined as global outside the function)
    """
    
    sce = str(int(100*w[0])) + '-' + str(int(100*w[1])) + '-' + str(int(100*w[2]))
    f_name = 'gep_results_' + 'T' + str(t) + '_' + 'W-' + sce + '.xlsx'
    
    df_d = pd.DataFrame(columns=["t", "g", "h", "d"])
    weighted_d = (weight_tj[:,None,None,:,None] * ( baseLoad + ((nEVs[:,None,None,None,None] * regional_EVLoad_)*(1/theta)))).sum(axis=(3,4))
    for t in range(len(tSet)):
        for g in range(len(subG)):
            for h in range(len(H)):
                df_d.loc[len(df_d)] = [ int(t), subG[g], H[h], weight_g[g] * weighted_d[t, g, h] ]
    
    with pd.ExcelWriter(f_name) as writer:
        df_s.to_excel(writer, sheet_name="s", index=False)
        df_ce.to_excel(writer, sheet_name="obj", index=False)
        df_v.to_excel(writer, sheet_name="v", index=False)
        df_y.to_excel(writer, sheet_name="y", index=False)
        df_u.to_excel(writer, sheet_name="u", index=False)
        df_d.to_excel(writer, sheet_name="d", index=False)
        df_slack_NC.to_excel(writer, sheet_name="slack_NC", index=False)
        df_NC_tr.to_excel(writer, sheet_name="NC_tr", index=False)
    
    results['Model'].append(sce)
    results['Cost'].append(df_ce['NPV'].sum())
    results['CO2'].append(df_ce['NetCO2'].sum())
    results['NOX'].append(df_ce['NetNOx'].sum())
    results['ObjVal'].append(gep.ObjVal)
    
# load_results FUNCTION
def load_results():
    
    """
    read results of all model results (.xlsx) existing in current working directory 
    into dataframes
        
    Parameters:
    None   
    
    Returns:
    dfs, dfce, dfv, dfu, dfd, df_slack_NC, df_NC_tr : dataframes for s, obj func variables, v, u, demand, regional new capacity transfer variables
    
    """
    
    dfs = pd.DataFrame(columns = ["cost", "CO2", "NOX", "t","b","k","s"])
    dfce = pd.DataFrame(columns=["cost", "CO2", "NOX", "t", 
                                 "IC", "MC", "PC", "UC", "TC", "CC","CS",
                                 "NetCost","NPV", 
                                 "GCE", "VCES", "NetCO2", 
                                 "GNE", "VNES", "NetNOx"])
    dfv = pd.DataFrame(columns = ["cost", "CO2", "NOX", "t","g","h","k","v"])
    dfy = pd.DataFrame(columns = ["t", "g", "h", "b0", "b1", "y"])
    dfu = pd.DataFrame(columns = ["cost", "CO2", "NOX", "t","g","h","u"]) 
    dfd = pd.DataFrame(columns = ["cost", "CO2", "NOX", "t","g","h","d"])
    df_slack_NC = pd.DataFrame(columns = ["t_strt", "t_end", "b", "NC", "slack", "dual", "slack_to_other_region", "dual_to_other_region", "slack_to_other_twindow", "dual_to_other_twindow"])
    df_NC_tr = pd.DataFrame(columns=["t", "tt", "b", "bb", "NC_tr"])
    
    # read all .xlsx file names that exist in current working directory
    files = []
    for file_path in os.listdir('.'):
        if os.path.isfile(os.path.join('.', file_path)):
            files.append(file_path)
            
    # loop through all these files
    for f_name in files:
        if f_name.startswith("gep_results_") and '.xlsx' in f_name:
            print(f"loading {f_name} results")
            
            t = int(f_name[f_name.find('_T')+2 : f_name.find('_W')])
            txt = f_name[f_name.find('_W')+3 : f_name.find('.xlsx')]
            w = txt.split('-')
            w = [round(int(w_)/100,2) for w_ in w]
            
            df = pd.read_excel(f_name,sheet_name='s')
            df['cost'] = w[0]
            df['CO2'] = w[1]
            df['NOX'] = w[2]
            dfs = pd.concat([dfs, df])
            
            df = pd.read_excel(f_name,sheet_name='obj')
            df['cost'] = w[0]
            df['CO2'] = w[1]
            df['NOX'] = w[2]
            dfce = pd.concat([dfce, df])
            
            
            df = pd.read_excel(f_name,sheet_name='v')
            df = df[df['v'] > 0]
            df['cost'] = w[0]
            df['CO2'] = w[1]
            df['NOX'] = w[2]
            dfv = pd.concat([dfv, df])
            
            df = pd.read_excel(f_name,sheet_name='y')
            df = df[df['y'] > 0]
            df['cost'] = w[0]
            df['CO2'] = w[1]
            df['NOX'] = w[2]
            dfy = pd.concat([dfy, df])

            
            df = pd.read_excel(f_name,sheet_name='u')
            df['cost'] = w[0]
            df['CO2'] = w[1]
            df['NOX'] = w[2]
            dfu = pd.concat([dfu, df])
            
            
            df = pd.read_excel(f_name,sheet_name='d')
            df['cost'] = w[0]
            df['CO2'] = w[1]
            df['NOX'] = w[2]
            dfd = pd.concat([dfd, df])
        
            
            df = pd.read_excel(f_name,sheet_name='slack_NC')
            df['cost'] = w[0]
            df['CO2'] = w[1]
            df['NOX'] = w[2]
            df_slack_NC = pd.concat([df_slack_NC, df])
            
            df = pd.read_excel(f_name,sheet_name='NC_tr')
            df['cost'] = w[0]
            df['CO2'] = w[1]
            df['NOX'] = w[2]
            df_NC_tr = pd.concat([df_NC_tr, df])
            
            print(f'finished scenario T{t} {w}')
    
    dfs = dfs.reset_index(drop=True)
    dfs['w'] = [str(int(100*dfs.loc[i]['cost'])) + ' ' + str(int(100*dfs.loc[i]['CO2'])) + ' ' + str(int(100*dfs.loc[i]['NOX'])) for i in dfs.index]

    dfce = dfce.reset_index(drop=True)
    dfce['w'] = [str(int(100*dfce.loc[i]['cost'])) + ' ' + str(int(100*dfce.loc[i]['CO2'])) + ' ' + str(int(100*dfce.loc[i]['NOX'])) for i in dfce.index]
    
    dfv = dfv.reset_index(drop=True)
    dfv['w'] = [str(int(100*dfv.loc[i]['cost'])) + ' ' + str(int(100*dfv.loc[i]['CO2'])) + ' ' + str(int(100*dfv.loc[i]['NOX'])) for i in dfv.index]

    dfy = dfy.reset_index(drop=True)
    dfy['w'] = [str(int(100*dfy.loc[i]['cost'])) + ' ' + str(int(100*dfy.loc[i]['CO2'])) + ' ' + str(int(100*dfy.loc[i]['NOX'])) for i in dfy.index]
    
    dfu = dfu.reset_index(drop=True)
    dfu['w'] = [str(int(100*dfu.loc[i]['cost'])) + ' ' + str(int(100*dfu.loc[i]['CO2'])) + ' ' + str(int(100*dfu.loc[i]['NOX'])) for i in dfu.index]

    dfd = dfd.reset_index(drop=True)
    dfd['w'] = [str(int(100*dfd.loc[i]['cost'])) + ' ' + str(int(100*dfd.loc[i]['CO2'])) + ' ' + str(int(100*dfd.loc[i]['NOX'])) for i in dfd.index]
    
    df_slack_NC = df_slack_NC.reset_index(drop=True)
    df_slack_NC['w'] = [str(int(100*df_slack_NC.loc[i]['cost'])) + ' ' + str(int(100*df_slack_NC.loc[i]['CO2'])) + ' ' + str(int(100*df_slack_NC.loc[i]['NOX'])) for i in df_slack_NC.index]
    
    df_NC_tr = df_NC_tr.reset_index(drop=True)
    df_NC_tr['w'] = [str(int(100*df_NC_tr.loc[i]['cost'])) + ' ' + str(int(100*df_NC_tr.loc[i]['CO2'])) + ' ' + str(int(100*df_NC_tr.loc[i]['NOX'])) for i in df_NC_tr.index]
    
    return dfs, dfce, dfv, dfu, dfd, df_slack_NC, df_NC_tr

def save_summary_results():
    
    results_df = pd.DataFrame.from_dict(results)
    w_cost_list = []
    w_co2_list = []
    w_nox_list = [] 
    for i in range(len(results_df)):
        w_list = results_df.iloc[i]["Model"].split("-")
        w_cost_list.append(float(w_list[0])/100)
        w_co2_list.append(float(w_list[1])/100)
        w_nox_list.append(float(w_list[2])/100)
        
    minObj1, maxObj1 = min(results['Cost']), max(results['Cost'])
    minObj2, maxObj2 = min(results['CO2']), max(results['CO2'])
    minObj3, maxObj3 = min(results['NOX']), max(results['NOX'])
    
    results_df['w_cost'] = w_cost_list
    results_df['w_co2'] = w_co2_list
    results_df['w_nox'] = w_nox_list
    
    results_df['Cost_'] = (results_df['Cost'] - minObj1) / (maxObj1 - minObj1)
    results_df['CO2_'] = (results_df['CO2'] - minObj2) / (maxObj2 - minObj2)
    results_df['NOX_'] = (results_df['NOX'] - minObj3) / (maxObj3 - minObj3)
    
    results_df['Obj_'] = results_df['w_cost'] * results_df['Cost_'] + results_df['w_co2'] * results_df['CO2_'] + results_df['w_nox'] * results_df['NOX_']
    
    results_df.to_excel('results.xlsx')

#%% Define Input Data Source File

# should exist under the same folder of this .py file (current working directory)
source_f = 'GEP_input_LP_basic.xlsx'

#%% Initial Used Memory

# to track the RAM usage during model difinition and solution process
memory = psutil.virtual_memory()
init_available_memory = memory.available / (1024 ** 3)  # Available memory in GB
print(f"Init Available Memory: {init_available_memory:.2f} GB")

#%% LOAD INPUT DATA

strt = perf_counter() # to track the elapsed time during input data loading 

# Sets and indices
'''
t   : years, t∈T={0,1,2,….Tmax} 
g   : days, g∈G={g1,g2,g3,…..} 
h   : hours,  h∈H= {0,1,…,23}  
j   : representative EV Load curves j∈J={j1,j2,j3,…..}
k   : generating unit types, k∈K={k1,k2,k3,….}
c   : EV charger types, c∈C={AC_L1,AC_L2,DC_Low,DC_High}
b   : regions, b∈B={b1,b2,…,b9}
A_b : set of neighboring regions of region b
      A[b1,b2] = 1 if b1 and b2 are neighbors and 0 otherwise
i   : objective criteria, i∈I={TotalNetCost ,CO2 emission, NOX emission}
'''
T = list(pd.read_excel(source_f,sheet_name='years')['t'])
G = list(pd.read_excel(source_f,sheet_name='days')['g'])
H = list(pd.read_excel(source_f,sheet_name='hours')['h'])
J = list(pd.read_excel(source_f,sheet_name='J')['j'])
B = list(pd.read_excel(source_f,sheet_name='regions_')['b'])

df = pd.read_excel(source_f,sheet_name='regions_',index_col='b')
A = {b: [] for b in B}
for b0 in B:
    for b1 in B:
        if df.loc[b0][b1] == 1:
            A[b0].append(b1)

K = list(pd.read_excel(source_f,sheet_name='unitTypes')['k'])
C = list(pd.read_excel(source_f,sheet_name='chargerTypes')['c'])

# generating unit types in K set
'''
    0: 'coal', 
    1: 'naturalGas', 
    2: 'bioMass', 
    3: 'geoThermal', 
    4: 'hydroRiver', 
    5: 'hydroDam', 
    6: 'wind', 
    7: 'solar', 
    8: 'nuclear'
'''

# Scalar Parameters
'''
r : interest rate (for calculating NPV)
FC : average cost of fuel (gasoline, diesel, LPG) ($/km)
cre : consumption rate (efficiency) of EV car (kW/km)
DD : yearly average driving distance of a car (km/year)
init_nCars : number of total cars at year zero
rCars : yearly increase rate in total number of cars
CarCO2 : CO2 emission per ICE car per km (tonCO2/km)
CarNOX : NOx emission per ICE car per km (tonNOx/km)
rBase : yearly increase rate in base load
UC : Unmet Demand Cost
LR : Loss Rate in Transmission
'''
df=pd.read_excel(source_f,sheet_name='main')[['parameter','value']].set_index('parameter')
r = df.loc['r']['value']
#init_nCars = df.loc['init_nCars']['value']
#rCars = df.loc['rCars']['value']
rBase = df.loc['rBase']['value']
DD = df.loc['DD']['value']
FC = 1e-6 * df.loc['FC']['value']
cre = df.loc['cre']['value']
CarCO2 = 1e-3 * df.loc['CarCO2']['value']
CarNOX = 1e-3 * df.loc['CarNOX']['value']
UC = 1e-3 * df.loc['UC']['value']
LR_tr = df.loc['LR_tr']['value']
LR_ds = df.loc['LR_ds']['value']
theta = 10000

# *****************************************************************************
# Special Settings for the problem instance

# -----------------------------------------------------------------------------
# Selected Representative Days
subG = G # use all represantative days

# -----------------------------------------------------------------------------
# Selected scenarios (Rep EV Load Curves)
J = J[:6] # use first 6 cha settings

# -----------------------------------------------------------------------------
# Selected subset of the planning period including t = 0 
maxT = 15
tSet = T[ : maxT + 1]

# -----------------------------------------------------------------------------
# Selected subset of new generating unit types that can be invested
subK = ['coal', 'naturalGas', 'wind', 'solar']

# -----------------------------------------------------------------------------
# Increase in Production Cost in Natural Gas Power Plants
incNatGasPC = 0.25

# Increase in New Connectable Transmission Capacity
nc_inc = 0.1

# -----------------------------------------------------------------------------
# Solar transmission cost reduction rate by region
#solar_regions = {b: 0.25 for b in B}
solar_regions = {b: 0 for b in B}

# Wind transmission cost increase rate by region
#wind_regions = {"b1": 1.25}
wind_regions = {b: 1 for b in B}

# -----------------------------------------------------------------------------
# EV Growth Scenario (EV-Low, EV-High)
EV_sce = "EV-Low"

# -----------------------------------------------------------------------------
# CHA Scenario (CHA-Dest, CHA-Balanced)
CHA_sce = "CHA-Dest"

# *****************************************************************************

# interest rate coefficient for each year (for NPV calculation)
r_coeff = [((1+r)**(-t)) for t in tSet]

# nEVx[t] : number of EVs in year t under selected EV adoption rate scenario
# E_t
df = pd.read_excel(source_f,sheet_name='nEVs')
nEVs = df[EV_sce].to_numpy().round(0)[:len(tSet)]
nCars = df["nCars"].to_numpy().round(0)[:len(tSet)]

# Weight of EV CHA scenario j in year t (each year weights sum up to 1)
# lower case omega_tj
df = pd.read_excel(source_f,sheet_name='weight(t,j)')
df = df[df['CHA_sce'] == CHA_sce]
weight_tj = np.zeros((len(tSet), len(J)))
temp = {(df.loc[i]['t'], j): df.loc[i][j] for j in J for i in df.index}
for j in J:
    for t in tSet:
        weight_tj [t,J.index(j)] = temp[t, j]
weight_tj = weight_tj.round(3) 

# parameters for new Charger Cost
df = pd.read_excel(source_f,sheet_name='chrg_dens_cost(t)')

newChgCost_ = 1e-6 * df['cost_new_Chrg'].to_numpy()
newChgCost_ = newChgCost_.round(3)[:len(tSet)]

lamda_tH = df['EV_per_charger_Home'].to_numpy()
lamda_tD = df['EV_per_charger_Public'].to_numpy()
lamda_tP = df['EV_per_charger_Public'].to_numpy()

w_chrg_cost_home = df['w_chrg_cost_home'].to_numpy()
w_chrg_cost_dest = df['w_chrg_cost_dest'].to_numpy()
w_chrg_cost_pcs = df['w_chrg_cost_pcs'].to_numpy()

df = pd.read_excel(source_f,sheet_name='chrg_dist(j)')
df = df[df['j'].isin(J)]
P_chrg = {}
for i in df.index:
    P_chrg[("H", J.index(df.loc[i]['j']))] = df.loc[i]['pHomeChrg']
    P_chrg[("D", J.index(df.loc[i]['j']))] = df.loc[i]['pDestChrg']

weighted_pchrg_home = np.zeros(len(tSet))
weighted_pchrg_dest = np.zeros(len(tSet))
for t in tSet:
     weighted_pchrg_home[t] = sum([weight_tj[t,j] * P_chrg[("H",j)] for j in range(len(J))])
     weighted_pchrg_dest[t] = sum([weight_tj[t,j] * P_chrg[("D",j)] for j in range(len(J))])

CC_H = np.zeros(len(tSet))
CC_D = np.zeros(len(tSet))
CC_P = np.zeros(len(tSet))
for t in tSet[1:]:
    CC_H[t] = w_chrg_cost_home[t] * ( (nEVs[t] / lamda_tH[t]) * weighted_pchrg_home[t] - (nEVs[t-1] / lamda_tH[t-1]) * weighted_pchrg_home[t-1])
    CC_D[t] = w_chrg_cost_dest[t] * ( (nEVs[t] / lamda_tD[t]) * weighted_pchrg_dest[t] - (nEVs[t-1] / lamda_tD[t-1]) * weighted_pchrg_dest[t-1])
    CC_P[t] = w_chrg_cost_pcs[t] * ( (nEVs[t] / lamda_tD[t]) * (1-weighted_pchrg_dest[t]) - (nEVs[t-1] / lamda_tD[t-1]) * (1-weighted_pchrg_dest[t-1]))

newChgCost = 1e-6 * np.array([ CC_H[t] + CC_D[t] + CC_P[t] for t in tSet])
#newChgCost = newChgCost.round(3)

# LT[k] : Lifetime of k
df = pd.read_excel(source_f,sheet_name='unitTypes')
LT = {}
for i in df.index:
    LT[df.loc[i]['k']] = df.loc[i]['LT']

# IC [t,k] : investment cost for unit type k in year t (millon $/ GW) -> CAPEX
# MC [t,k] : operations and maintanence cost for unit type k in year t (million $/GW-year) -> Fixed O&M Costs
# PC [t,k] : production cost (million $/GW) for unit type k in year t -> Variable Costs
df = pd.read_excel(source_f,sheet_name='unitCosts(t,k)_')
df = df[df['k'].isin(K)] # filter only the allowed unit types 

IC = 1e-3 * df.pivot(index='t', columns='k', values='investCost_').reindex(columns=K).fillna(0).to_numpy() # conversion from USD/MW to millon USD/GW
for t in T:
    IC[t,K.index('nuclear')] = 0 # neglect the investment cost for nuclear power plant    
IC = IC[:len(tSet),:]
#IC = IC.round(3)

MC = 1e-3 * df.pivot(index='t', columns='k', values='opmaintCost_').reindex(columns=K).fillna(0).to_numpy()
MC = MC[:len(tSet),:]
#MC = MC.round(3)

PC = 1e-3 * df.pivot(index='t', columns='k', values='prodCost_').reindex(columns=K).fillna(0).to_numpy()
PC = PC[:len(tSet),:]
#PC = PC.round(6)

# reflect the increase in Natural Gas Prices
if incNatGasPC > 0:
    for t in tSet:
        PC[t,K.index('naturalGas')] = (1 + incNatGasPC) * PC[t,K.index('naturalGas')]

# IB [t] : investment cost budget in year t (million $)
df = pd.read_excel(source_f,sheet_name='budget')
IB = 1e-6 * df['yearlyTotal'].to_numpy() # with conversion from USD to million USD
#IB = IB.round(3) #or round to 3 digits

# FTC [b] : Fixed Transmission Connection Cost per GW capacity per year (million $ / GW - year)
# VTC [b] : Variable Transmission Connection Cost (million $ / GWh)
df = pd.read_excel(source_f,sheet_name='transmissionCosts').set_index('b')
FTC = 1e-3 * df['FTC_Gen ($/MW-year)'].to_numpy() # with conversion from USD/MW to million USD/GW
#FTC = FTC.round(4) #or round to 4 digits
VTC = 1e-3 * df['VTC_Gen ($/MWh)'].to_numpy() # with conversion from USD/MW to million USD/GW
#VTC = VTC.round(6) #or round to 6 digits

# decrease or inclease in transmission cost for some sources/regions
# decrese: direct connection to distribution network
# increase: regional feasibility considerations
TC_ = np.zeros((len(B), len(K)))
for b in B:
    for k in K:
        if k == 'solar':
            if b in solar_regions:
                TC_[B.index(b),K.index(k)] = solar_regions[b]  # solar transmission cost is cheaper in all regions 
            else:
                TC_[B.index(b),K.index(k)] = 1
        elif k == 'wind':
            if b in wind_regions:
                TC_[B.index(b),K.index(k)] = wind_regions[b]   # wind transmission cost is more costly in certain regions 
            else:
                TC_[B.index(b),K.index(k)] = 1
        else:
            TC_[B.index(b),K.index(k)] = 1

# initCap [b,k] : initial installed capacity of unit type k in region b in year 0 (GW)
df = pd.read_excel(source_f,sheet_name='initCapacity(k,b)_')
df = df[df['k'].isin(K)]

temp = { (b, df.loc[i]['k']): df.loc[i][b] for b in B for i in df.index}
initCap = np.zeros((len(B), len(K)))
b_idx = 0
for b in B:
    k_idx = 0
    for k in K:
        initCap[b_idx,k_idx] = temp[(b,k)] * 1e-3 # with conversion from MW to GW
        k_idx += 1
    b_idx += 1
#initCap = initCap.round(3)

# max capacity that can be added for unit type k
maxCap = {k: 10 for k in K}     

# NC [t,b] : new connectable capacity (in GW) upper bound in region b (from year 0 up to year t)
df = pd.read_excel(source_f,sheet_name='NewCap(t,b)_')
NC = {}
for i in df.index:
    NC[(df.loc[i]['t'], df.loc[i]['b'])] = 1e-3 * df.loc[i]['connectable_new_cap'] # with conversion from MW to GW
    NC[(df.loc[i]['t'], df.loc[i]['b'])] = NC[(df.loc[i]['t'], df.loc[i]['b'])].round(3) # round to 3 digits
nc_t_b = list(NC.keys())

NC_original = {}
for i in df.index:
    NC_original[(df.loc[i]['t'], df.loc[i]['b'])] = 1e-3 * df.loc[i]['connectable_new_cap'] # with conversion from MW to GW
    NC_original[(df.loc[i]['t'], df.loc[i]['b'])] = NC_original[(df.loc[i]['t'], df.loc[i]['b'])].round(3) # round to 3 digits
nc_t_b = list(NC.keys())

# CE [k]: CO2 emission of generating unit type k (tonCO2/GWh)
# NE [k]: NOx emission of generating unit type k (tonNOx/GWh)
df = pd.read_excel(source_f,sheet_name='emissions').set_index('k').reindex(K).fillna(0)
df = df[df.index.isin(K)]
CE = df['CO2'].to_numpy() # no conversion needed for kgCO2/MWh --> tonCO2/GWh
NE = df['NOx'].to_numpy() # no conversion needed for kgCO2/MWh --> tonCO2/GWh

# hourly ramp-rate coefficients ρₖ 
# fraction of installed capacity that each technology can increase / decrease within one hour
rho = {
    'coal'        : 0.08,
    'naturalGas'  : 0.25,
    'bioMass'     : 0.12
}

# cf [g,h,b,k] : capapacity factor of unit type k in region b on load_curve (day) g and hour h
df = pd.read_excel(source_f,sheet_name='cf(k,g,h)')
df = df[df['k'].isin(K)]
df = df[df['g'].isin(subG)]
cf_ghk = {(df.loc[i]['g'], hr, df.loc[i]['k']): df.loc[i][hr] for hr in H for i in df.index}

df = pd.read_excel(source_f,sheet_name='cf(k,b)')
df = df[df['k'].isin(K)]
cf_kb = {(df.loc[i]['k'], b): df.loc[i][b] for b in B for i in df.index}

cf = np.zeros((len(subG), len(H), len(B), len(K)))

for g_idx, g in enumerate(subG):
    for hr in H:
        for b_idx, b in enumerate(B):
            for k_idx, k in enumerate(K):
                cf[g_idx, H.index(hr), b_idx, k_idx] = cf_ghk[g, hr, k] * cf_kb[k, b]

cf = cf.round(3)

# Demand (baseload + EV load) _________________________________________________

# baseLoad per (g, h): hourly loads per each representative day g
df = pd.read_excel(source_f,sheet_name='baseLoad(g)')
df = df[df['g'].isin(subG)]
temp = {(df.loc[i]['g'],hr): df.loc[i][hr] for hr in H for i in df.index}
baseLoad_ = np.zeros((len(subG), len(H)))
for g_idx, g in enumerate(df['g'].unique()):
    for hr in H:
        baseLoad_[g_idx, H.index(hr)] = temp[(g, hr)]
del temp

# baseLoad weights per representative day g (weights sum up to 365)
df = pd.read_excel(source_f,sheet_name='baseLoad(g)')
if len(subG) == len(G):
    weight_g = df['weight'].to_numpy()
else:
    # if only a subset of repdays are used then rearrange the weights
    unselected_weight = df[~df['g'].isin(subG)]['weight'].sum()
    selected_weight = df[df['g'].isin(subG)]['weight'].sum()
    df = df[df['g'].isin(subG)]
    weight_g = df['weight'].to_numpy()
    for g in range(len(weight_g)):
        weight_g[g] = weight_g[g] + unselected_weight * (weight_g[g] / selected_weight)

# baseLoad weights per (g,b): consumption share of each region b on representative day g 
# (for each day g weights of regions sum up to 1)
df = pd.read_excel(source_f,sheet_name='baseLoadWeight(g,b)')
df = df[df['g'].isin(subG)].set_index('g')
baseload_weight_g_b = df.to_numpy()[:,:len(B)] 

# baseLoad per (t,g,h,b) (each year t's baseload is icreased by rBase rate)
baseLoad = np.zeros((len(tSet), len(subG), len(H), len(J), len(B)))
for t in range(len(tSet)):
    for g in range(len(subG)):
        for h in range(len(H)):
            for j in range(len(J)):
                for b in range(len(B)):
                    baseLoad[t,g,h,j,b] = 1e-3 * ((1+rBase)**t) * baseload_weight_g_b[g,b] * baseLoad_[g,h]

# EVLoad weights per (g,b): EV Load share of each region b on representative day g 
# (for each day g weights of regions sum up to 1)
df = pd.read_excel(source_f,sheet_name='EVLoadWeight(g,b)').set_index('g')
EVLoad_weight_g_b = df.to_numpy()[:len(subG),:]

# EVLoad_(g,h,j): hourly EV charging loads on representative day g with charger availability scenario (environment setting) j
# regioal_EVLoad_(g,h,j,b): hourly EV charging loads on representative day g with cha setting j in region b
df = pd.read_excel(source_f,sheet_name='10 000 EVLoad(g,j)')
df = df[df['g'].isin(subG)]
temp = {(df.loc[i]['g'], df.loc[i]['j'], hr): df.loc[i][hr] for hr in H for i in df.index}
EVLoad_ = np.zeros((len(subG), len(H), len(J)))
regional_EVLoad_ = np.zeros((len(subG), len(H), len(J), len(B)))
for g in subG:
    for hr in H:
        for j in J:
            EVLoad_[subG.index(g), H.index(hr), J.index(j)] = temp[(g, j, hr)]
            for b in B:
                regional_EVLoad_[subG.index(g), H.index(hr), J.index(j), B.index(b)] = 1e-3 * EVLoad_weight_g_b[subG.index(g), B.index(b)] * EVLoad_[subG.index(g), H.index(hr), J.index(j)]
del temp

# Weight of EV CHA scenario j in year t (each year weights sum up to 1)
df = pd.read_excel(source_f,sheet_name='weight(t,j)')
weight_tj = np.zeros((len(tSet), len(J)))
temp = {(df.loc[i]['t'], j): df.loc[i][j] for j in J for i in df.index}
for j in J:
    for t in tSet:
        weight_tj [t,J.index(j)] = temp[t, j]
del(temp)

# Calculate the Average Driving Distance per EV (DD) based on representative EV Loads
'''
Explanation of DD calculation
365 : days
1000 * (1/cre) : km per MWh of energy consumption
np.sum([...])/10000 : Total daily EV load devided by the nEVs (10 000)
'''
cre_MWh_per_km = cre/1000  # conversion from kWh/km to MWh/km
DD_percar_gj = (EVLoad_.sum(axis=1) / cre_MWh_per_km) / 10000
DD = 0
for g_idx, g in enumerate(subG):
    for j_idx, j in enumerate(J):
        DD += weight_g[g_idx] * (weight_tj.sum(axis=0)/len(tSet))[j_idx] *  DD_percar_gj[g_idx, j_idx]
DD = round(DD,0)

'''
# EVLoad[t,g,h,j,b]: EV charging load weighted by g,b and scaled to nEVs[t]
EVLoad = np.zeros((len(tSet), len(subG), len(H), len(J), len(B)))
for t in range(len(tSet)):
    for b in range(len(B)):
        for g in range(len(subG)):
            for j in range(len(J)):
                for h in range(len(H)):
                    EVLoad[t,g,h,j,b] = (nEVs[t]/theta) * EVLoad_weight_g_b[g,b] * 1e-3 * EVLoad_[g,h,j]

# Total Demand (baseload + EVLoad)
# d[t,g,h,j,b] : electricity demand of region b in year t day g hour h with scenario j 
d = np.zeros((len(tSet), len(subG), len(H), len(J), len(B)))
for t in range(len(tSet)):
    for g in range(len(subG)):
        for h in range(len(H)):
            for j in range(len(J)):
                for b in range(len(B)):
                    d[t,g,h,j,b] = (baseLoad[t,g,h,j,b] + EVLoad[t,g,h,j,b])

    
# alpha[t,g,j] : weight of scenario j on day j in year t
alpha = np.zeros((len(tSet), len(subG), len(J)))
for t in range(len(tSet)):
    for g in range(len(subG)):
        for j in range(len(J)):
            alpha[t,g,j] = weight_g[g] * weight_tj[t,j]   
'''

# track RAM usage and elapsed time during inputdata loading 
memory = psutil.virtual_memory()
print(f'Input Data Loaded... ({(init_available_memory - memory.available / (1024 ** 3) ):.2f} GB) Loaded... in {round((perf_counter()-strt)/60, 1)} minutes')

#%% Define Gurobi licence papameters and environment
gurobiparams = {
"WLSACCESSID": '3ab6af7f-4081-4dd9-91c5-f1cb244644a9',
"WLSSECRET": 'b62dc636-5e78-4cd8-8e9e-d507bc2ede2e',
"LICENSEID": 2458030, 
}
env = gp.Env(params=gurobiparams)

#%% Define Model (Variables and Constraints)

strt = perf_counter() # reset elapsed-time

# define a new model with the gurobi environment parameters
gep = gp.Model(env=env)

n_timesteps = len(tSet) #time-steps (years)
n_regions = len(B) #regions
n_hours = len(H) #hours in a day
n_rep_days = len(subG) #representative base_load demand curves
n_scenarios = len(J) #representative EV_load curves
n_units = len(K) #generating unit types
n_subunits = len(subK)

# Variables ___________________________________________________________________
''' 
s[t,b,k] : new capacity (GW) of type k in year t in region b (positive)
v[t,g,h,j,b,k] : amount of electricity (MWh) to be generated in year t in day g repcurve j at h in region b in unit type k  
y[t,g,h,j,b,bb] : amount of electricity (MWh) to be transmitted from b to bb in year t in day g repcurve j at h
u[t,g,h,j,b] : unmet demand in year t day g repcurve j at hour h in region b
NC_reg_tr[t, tt, b, bb] : regional connectable new capacity of region b in timewindow ending with t to be transferred region bb in timewindow ending with t 
'''
s = gep.addMVar((n_timesteps, n_regions, n_subunits), lb=0.0, name="s") # investment decision
v = gep.addMVar((n_timesteps, n_rep_days, n_hours, n_scenarios, n_regions, n_units), vtype=GRB.CONTINUOUS, lb=0.0, name="v") # generation decision
y = gep.addMVar((n_timesteps, n_rep_days, n_hours, n_scenarios, n_regions, n_regions), vtype=GRB.CONTINUOUS, lb=0.0, name="y") # transmission decision
u = gep.addMVar((n_timesteps, n_rep_days, n_hours, n_scenarios, n_regions), vtype=GRB.CONTINUOUS, lb=0.0, name="u") # unmet demand

NC_reg_tr= gep.addMVar((n_timesteps, n_timesteps, n_regions, n_regions), lb=0.0, name="NC_reg_tr") # regional new capacity transfer decision

'''
CAP[t,b,k] : installed capacity (GW) of type k unit in year t in region b 
TIC[t] Total investment Cost of new generation units in year t
TMC[t] Total maintanence Cost of generation units in year t
TFTC[t] Total Fixed transmission Cost in year t
TPC[t] Total production Cost of generation in year t
TUC[t] Total unmet demand Cost in year t
TCC[t] Total charger Cost for new EVs in year t
TVTC[t] Total transmission Cost in year t
CS[t]  Cost Savings due to unused gasoline in year t
GCE[t] Generation CO2 emissions in year t
VCES[t] Vehicle Carbon Emission Savings due to EVs in year t
GNE[t] Generation NOx emission in year t
SV total salvage value
'''
# Auxillary Variables
CAP = gep.addMVar((n_timesteps, n_regions, n_units), name="CAP") # installed capacity
TIC = gep.addMVar(n_timesteps, lb=0, ub=IB[:n_timesteps], name="TIC" ) # Total Investment Cost
TMC = gep.addMVar(n_timesteps, lb=0, name="TMC")  # Total Maintenance Cost
TFTC = gep.addMVar(n_timesteps, lb=0, name="TFTC")  # Total Fixed Transmission Cost
TPC = gep.addMVar(n_timesteps, lb=0, name="TPC")  # Total Production Cost
TUC = gep.addMVar(n_timesteps, lb=0, name="TUC")  # Total Unmet Demand Cost
TVTC = gep.addMVar(n_timesteps, lb=0, name="TVTC")  # Total Variable Transmission Cost
TCC = gep.addMVar(n_timesteps, lb=0, name="TCC")  # Total Charger Cost
CS = gep.addMVar(n_timesteps, name="CS")    # Cost Savings due to unused vehicle fuel
GCE = gep.addMVar(n_timesteps, name="GCE")  # Generated CO2 Emissions
VCES = gep.addMVar(n_timesteps, name="VCES")  # Vehicle CO2 Emission Savings
GNE = gep.addMVar(n_timesteps, name="GNE")  # Generated NOX Emissions
VNES = gep.addMVar(n_timesteps, name="VNES")  # Vehicle NOX Emission Savings
SV = gep.addVar(lb=0, name="SV")

# track RAM usage and elapsed time during variable definition 
gep.update()
memory = psutil.virtual_memory()
print(f'Variables Defined... ({(init_available_memory - memory.available / (1024 ** 3)):.2f} GB) Loaded... in {round((perf_counter()-strt)/60, 1)} minutes')
  

# Constraints _________________________________________________________________

strt = perf_counter() # reset elapsed-time

# Auxillary Constraints for Calculating Cumulative Capacities at year t (CAP[t,b,k])

Const_InitCap = gep.addConstrs((
    CAP[0,b,k] == initCap[b,k]
    for b in range(len(B)) for k in range(len(K)) ), name='initCap_' )
        
for k_idx, k in enumerate(K):
    if k in subK:
        gep.addConstrs(( CAP[t,b,k_idx] - CAP[t-1, b, k_idx] - s[t, b, subK.index(k)] == 0
                        for t in tSet[1:] for b in range(len(B)) for kk in [k_idx] ), name="CAP_");
    else:
        gep.addConstrs(( CAP[t,b,k_idx] == initCap[b,k_idx] for t in tSet[1:] for b in range(len(B)) for kk in [k_idx] ), name="CAP_" );


# Auxillary Constraints for Calculating Cost Variables
 
# Investment Cost, TIC[t] (Matrix-Based)
sub_IC = np.zeros((n_timesteps, len(subK)))
for t in tSet:
    for k_idx, k in enumerate(K):
        if k in subK:
            sub_IC[t,subK.index(k)] = IC[t,k_idx]

Const_TIC = gep.addConstrs((
    TIC[t] == (sub_IC[t, :] @ s[t, :, :].sum(axis=0)) 
    for t in tSet), name="TIC_");

# Maintanence Cost, TMC[t] (Matrix-Based)
Const_TMC = gep.addConstrs((
    TMC[t] == (MC[t, :] @ CAP[t, :, :].sum(axis=0)) 
    for t in tSet), name="TMC_");

# Fixed Transmission Cost, TFTC[t] (Matrix-Based)
FTC_ = FTC[:, None] * TC_  # shape(bk)
Const_TFTC = gep.addConstrs((
    TFTC[t] == ((FTC_ * CAP[t, :, :]).sum())
    for t in tSet), name='TFTC_');

# Variable Transmission Cost, TVTC[t] (Matrix-Based)
#weighted_v = (v.sum(axis=2) * alpha[:, :, :, None, None]).sum(axis=(1, 2)) # shape (tbk)
weighted_v = ((v.sum(axis=2) * weight_g[None,:,None,None,None]) * weight_tj[:,None,:,None,None]) .sum(axis=(1, 2)) # shape (tbk)
VTC_ = VTC[:, None] * TC_  #shape(bk)
Const_TVTC = gep.addConstrs((
    TVTC[t] == ((VTC_ * weighted_v[t, :, :]).sum()) 
    for t in tSet), name="TVTC_");
   
# Production Cost, TPC[t] (Matrix-Based)
#weighted_v = (v.sum(axis=(2,4)) * alpha[:, :, :, None]).sum(axis=(1, 2)) #shape (tk)
weighted_v = ((v.sum(axis=(2,4)) * weight_g[None,:,None,None]) * weight_tj[:,None,:,None]).sum(axis=(1, 2)) #shape (tk)
Const_TPC = gep.addConstrs((
    TPC[t] == (PC[t, :] @ weighted_v[t, :]) 
    for t in tSet), name="TPC_");
    
# Unmet Demand Cost, TUC[t] (numpy array multiplication)
#weighted_u = (u.sum(axis=(2,4)) * alpha[:, :, :]).sum(axis=(1, 2))
weighted_u = ((u.sum(axis=(2,4)) * weight_g[None,:,None]) * weight_tj[:,None,:]).sum(axis=(1, 2))
Const_TUC = gep.addConstrs((
    TUC[t] == UC * weighted_u[t] for t in tSet), name="TUC_");

# Charger Cost, TCC[t] (Matrix-Based)
Const_TCC = gep.addConstrs((
    TCC[t] == newChgCost[t]
    for t in tSet[1:]), name="TCC_");

# Cost Savings, CS[t], due to Unused Fuel in ICE Vehicles (Matrix-Based)
Const_CS = gep.addConstrs((
    CS[t] == DD * FC * nEVs[t]
    for t in tSet), name="CS_");

# Salvage Value SV
Const_SV = gep.addConstr((
    SV == ((1+r)**(-n_timesteps)) * gp.quicksum(sub_IC[t, subK.index(k)] * s[t,b,subK.index(k)] * (LT[k] - (len(tSet) - t + 1))/LT[k] 
                                               for t in tSet for b in range(n_regions) for k in subK)), name='SV_')

# Auxillary Constraints for Calculating CO2 Emission Variables

#weighted_v = (v.sum(axis=(2,4)) * alpha[:, :, :, None]).sum(axis=(1, 2)) #shape (tk)
weighted_v = ((v.sum(axis=(2,4)) * weight_g[None,:,None,None]) * weight_tj[:,None,:,None]).sum(axis=(1, 2)) #shape (tk)

# CO2 Emission from Generators (Matrix-Based)
Const_GCE = gep.addConstrs((
    GCE[t] == (weighted_v[t, :] @ CE) 
    for t in tSet), name="GCE_");

# CO2 Emission Savings due to Reducing CO2 Emissions from Vehicles (Matrix-Based)
Const_VCES = gep.addConstrs((
    VCES[t] == DD * CarCO2 * nEVs[t]
    for t in tSet), name="VCES_");

# Auxillary Constraints for Calculating NOX Emission Variables (Matrix-Based)

# NOX Emission from Generators
Const_GNE = gep.addConstrs((
    GNE[t] == (weighted_v[t, :] @ NE) 
    for t in tSet), name="GNE_");

# NOX Emission Savings due to Reducing CO2 Emissions from Vehicles (Matrix-Based)
Const_VNES = gep.addConstrs((
    VNES[t] == DD * CarNOX * nEVs[t]
    for t in tSet), name="VNES_");

# Flow Balance Constraints and Available Capacity Constraints

A_ = np.zeros((n_regions, n_regions)) # construct the region neighborhood matrix
for b0 in B:
    for b1 in B:
        if b1 in A[b0] and b0 != b1:
            A_[B.index(b0), B.index(b1)] = 1 # if b0 and b1 are neighbors then set to 1

#LHS_flow = v.sum(axis=5) + (1 - LR_tr) * (A_[None, None, None, None,:,:] * y).sum(axis=4) - (A_[None, None, None, None,:,:] * y).sum(axis=5) + u - d/(1-LR_ds)
LHS_flow = v.sum(axis=5) + (1 - LR_tr) * (A_[None, None, None, None,:,:] * y).sum(axis=4) - (A_[None, None, None, None,:,:] * y).sum(axis=5) + u - (1/(1-LR_ds)) * ( baseLoad + ((nEVs[:,None,None,None,None] * regional_EVLoad_)*(1/theta)))

weighted_cap = cf[None, :, :, :, :] * CAP[:, None, None, :, :]
LHS_cap = (weighted_cap[:,:,:,None,:,:] - v)

for t in tSet:
    for g in range(len(subG)):
        for h in range(len(H)):
            for j in range(len(J)):
                for b in range(len(B)):
                    gep.addConstr(( LHS_flow[t,g,h,j,b] >= 0), name=f"Balance_{t}_{g}_{h}_{j}_{b}");
                    for k in range(len(K)):
                        gep.addConstr(( LHS_cap[t,g,h,j,b,k] >= 0), name=f"Capacity_{t}_{g}_{h}_{j}_{b}_{k}");

'''
# Constraints for hourly ramp-rate envelope for dispatchable technologies 
# Limitting how fast each generator’s dispatch (v) may rise or fall between two consecutive hours, 
# without introducing unit-commitment binaries.
for t in tSet:
    #print(f"started t-{t}")
    for g in range(len(subG)):
        for j in range(len(J)):
            for b_idx, b in enumerate(B):
                for k_idx, k in enumerate(K):
                    if k in rho:
                        
                        for h in range(1, len(H)):  # h = 1 … 23
                            # Ramp-up
                            gep.addConstr(
                                v[t,g,h,j,b_idx,k_idx] - v[t,g,h-1,j,b_idx,k_idx]
                                <= rho[k] * CAP[t,b_idx,k_idx],
                                name=f"RampUp_{t}_{g}_{h}_{j}_{b_idx}_{k_idx}")
                            
                            # Ramp-down
                            gep.addConstr(
                                v[t,g,h-1,j,b_idx,k_idx] - v[t,g,h,j,b_idx,k_idx]
                                <= rho[k] * CAP[t,b_idx,k_idx],
                                name=f"RampDn_{t}_{g}_{h}_{j}_{b_idx}_{k_idx}")                 
'''

# New connectable capacity constraints (new capacities can not exceed the allowed limits in each region)
for (t,b) in nc_t_b:
    NC[(t,b)] = (1 + nc_inc)* NC[(t,b)]

# fixed NC
'''
for (t,b) in NC:
    if t <= len(tSet):
        if "nuclear" in subK:
            gep.addConstr( (gp.quicksum( s[tt, B.index(b), k] for tt in tSet[:t+1] for k in range(n_subunits) if k != subK.index('nuclear')) <= NC[(t,b)]), name=f'new_connectable_cap_{t}_{B.index(b)}')
        else:
            gep.addConstr( (gp.quicksum( s[tt, B.index(b), k] for tt in tSet[:t+1] for k in range(n_subunits) ) <= NC[(t,b)]), name=f'new_connectable_cap_{t}_{B.index(b)}')
'''

# with NC transfer
t_list = []
for (t,b) in nc_t_b:
    if t not in t_list:
        t_list.append(t)
t_list.sort()

NC_ = {}
for t_index in range(len(t_list)):
    for b in B:
        if t_index == 0:
            NC_[(1, t_list[t_index], b)] = NC[(t_list[t_index], b)]
        else:
            NC_[(t_list[t_index-1] + 1, t_list[t_index], b)] = round(NC[(t_list[t_index],b)] - NC[(t_list[t_index-1],b)],3)

for (t_strt, t_end, b) in NC_:
    if t_end <= len(tSet):

        if "nuclear" in subK:
            gep.addConstr( (gp.quicksum( s[tt, B.index(b), k] for tt in range(t_strt, t_end + 1) for k in range(n_subunits) if k != subK.index('nuclear')) <= NC_[(t_strt, t_end, b)]), name=f'new_connectable_cap_{t_strt}_{t_end}_{B.index(b)}')
        else:
            gep.addConstr( (gp.quicksum( s[tt, B.index(b), k] for tt in range(t_strt, t_end + 1) for k in range(n_subunits) ) 
                            + gp.quicksum( NC_reg_tr[t_end, t_end, B.index(b), B.index(bb)] for bb in B if bb != b) 
                            - gp.quicksum( NC_reg_tr[t_end, t_end, B.index(bb), B.index(b)] for bb in B if bb != b)
                            + gp.quicksum( NC_reg_tr[t_end, tt, B.index(b), B.index(b)] for tt in t_list if tt <= len(tSet) if tt > t_end ) 
                            - gp.quicksum( NC_reg_tr[tt, t_end, B.index(b), B.index(b)] for tt in t_list if tt <= len(tSet) if tt < t_end )
                            <= NC_[(t_strt, t_end, b)]), name=f'new_connectable_cap_{t_strt}_{t_end}_{B.index(b)}')

# limit transfers (0: not allowed 1: fully allowed)
NC_tr_coeff_within_regions = 0
NC_tr_coeff_within_twindows = 0
for (t_strt, t_end, b) in NC_:
    if t_end in tSet:
        gep.addConstr( (gp.quicksum(NC_reg_tr[t_end, t_end, B.index(b), B.index(bb)] for bb in B if bb != b) <= NC_tr_coeff_within_regions * NC_[(t_strt, t_end, b)]), name=f'trans_connectable_cap_otherregions_{t_strt}_{t_end}_{B.index(b)}')
        gep.addConstr( (gp.quicksum(NC_reg_tr[t_end, tt, B.index(b), B.index(b)] for tt in t_list if tt <= len(tSet) if tt > t_end) <= NC_tr_coeff_within_twindows * NC_[(t_strt, t_end, b)]), name=f'trans_connectable_cap_othertwidows_{t_strt}_{t_end}_{B.index(b)}')

# Ensure that the unmet demand in the last year is not more than all previous years' average
gep.addConstr( (TUC[-1] <= gp.quicksum( TUC[t] for t in tSet[:-1])/len(tSet[:-1]) ), name='UD__')

# New investments are not allowed in t = 0 
gep.addConstrs((s[0,b,k] == abs(0) for b in range(len(B)) for k in range(len(subK)) ), name="unallowed_initialyear");

                 
# GEP model params ____________________________________________________________
gep.resetParams()
gep.Params.ScaleFlag = 3
gep.Params.BarHomogeneous = 1
gep.params.BarConvTol = 1e-5
gep.params.FeasibilityTol = 1e-6
gep.params.OptimalityTol = 1e-6
gep.params.BarIterLimit  = 1000

# track RAM usage and elapsed time during constraint definition 
gep.update()
memory = psutil.virtual_memory()
print(f'Constraints Defined... ({(init_available_memory - memory.available / (1024 ** 3) ):.2f} GB) Loaded... in {round((perf_counter()-strt)/60, 1)} minutes')

#%% initialize the results dictionary

results = {'iter':[], 'Model':[], 'Cost':[], 'CO2':[], 'NOX':[], 'ObjVal': [], 'SolTime':[]}
CO2_coeff, NOX_coeff = 0, 0

#%% Function for Solving the Model for all weight combinations

def solve_model(iter_n=0):
        
    # Solve only for min Cost _____________________________________________________
    strt = perf_counter()
    
    w = [1.0, 0.0, 0.0]
    obj = gp.quicksum( ((1+r)**(-t)) * (TIC[t] + TMC[t] + TPC[t] + TUC[t]+ TFTC[t] + TVTC[t] + TCC[t] - CS[t]) for t in range(n_timesteps) )-SV
    gep.setObjective(obj, GRB.MINIMIZE)
    gep.optimize()
    
    results['SolTime'].append( round( (perf_counter() - strt)/60 , 1))
    results['iter'].append(iter_n)
    
    df_s, df_ce, df_v, df_y, df_u, df_slack_NC, df_NC_tr = get_results()
    
    # unmet demand with minCost model
    u_mincost = df_u.groupby('t')['u'].sum().to_dict()
    if len(u_mincost) > 0:
        for t in tSet:
            if t not in u_mincost:
                u_mincost[t] = 0
    else:
        u_mincost = {t: 0 for t in tSet}
    
    cost_with_minCost = df_ce['NPV'].sum()
    CO2_with_minCost = df_ce['NetCO2'].sum()
    NOX_with_minCost = df_ce['NetNOx'].sum()
    
    save_results( maxT, w, df_s, df_ce, df_v, df_y, df_u, df_slack_NC, df_NC_tr)
    
    # solve for min CO2 _______________________________________________________
    strt = perf_counter()
    
    gep.reset()
    
    w = [0.0,  1.0, 0.0]
    
    # add a new constraint 
    u_cons = gep.addConstr( (gp.quicksum(r_coeff[t] * TUC[t] for t in tSet) == (1 + w[1])* gp.quicksum(r_coeff[t] * UC * u_mincost[t] for t in tSet)), name='UD_')
    gep.update()
    
    
    obj = gp.quicksum( GCE[t] - VCES[t] for t in tSet)
    gep.setObjective(obj, GRB.MINIMIZE)     
    gep.optimize()
    
    results['SolTime'].append( round( (perf_counter() - strt)/60 , 1))
    results['iter'].append(iter_n)
    
    df_s, df_ce, df_v, df_y, df_u, df_slack_NC, df_NC_tr = get_results()    
    save_results( maxT, w, df_s, df_ce, df_v, df_y, df_u, df_slack_NC, df_NC_tr )
    
    cost_with_minCO2 = df_ce['NPV'].sum()
    CO2_with_minCO2 = df_ce['NetCO2'].sum()
    
    # solve for min NOX _______________________________________________________
    strt = perf_counter()
    
    gep.reset()
    
    w = [0.0,  0.0, 1.0]
    
    constraint = gep.getConstrByName('UD_')
    if constraint is not None:
        gep.remove(constraint)
    gep.update()
    
    u_cons = gep.addConstr( (gp.quicksum(r_coeff[t] * TUC[t] for t in tSet) == (1 + w[2])* gp.quicksum(r_coeff[t] * UC * u_mincost[t] for t in tSet)), name='UD_')
    gep.update()
    
    obj = gp.quicksum( GNE[t] - VNES[t] for t in tSet)
    gep.setObjective(obj, GRB.MINIMIZE)
    gep.optimize()
         
    results['SolTime'].append( round( (perf_counter() - strt)/60 , 1))
    results['iter'].append(iter_n)
    
    df_s, df_ce, df_v, df_y, df_u, df_slack_NC, df_NC_tr = get_results()    
    save_results( maxT, w, df_s, df_ce, df_v, df_y, df_u, df_slack_NC, df_NC_tr )
    
    cost_with_minNOX = df_ce['NPV'].sum()
    NOX_with_minNOX = df_ce['NetNOx'].sum()
        
    # Calculate CO2 and NOX coeffs ____________________________________________
    # alternative-1
    cost_coeff = 1
    CO2_coeff = (cost_with_minCO2 - cost_with_minCost)/(CO2_with_minCost - CO2_with_minCO2) 
    NOX_coeff = (cost_with_minNOX - cost_with_minCost)/(NOX_with_minCost - NOX_with_minNOX)
        
    # alternative-2
    minObj1, maxObj1 = min(results['Cost']), max(results['Cost'])
    minObj2, maxObj2 = min(results['CO2']), max(results['CO2'])
    minObj3, maxObj3 = min(results['NOX']), max(results['NOX'])
    
    cost_coeff = 1
    CO2_coeff = (maxObj1 - minObj1)/(maxObj2 - minObj2)
    NOX_coeff = (maxObj1 - minObj1)/(maxObj3 - minObj3)

    # Solve for each mixed obj weight conbination in a loop ___________________
    for w in objWeights_list:
        strt = perf_counter()
        
        gep.reset()
        
        constraint = gep.getConstrByName('UD_')
        if constraint is not None:
            gep.remove(constraint)
        gep.update()
        
        u_cons = gep.addConstr( (gp.quicksum(r_coeff[t] * TUC[t] for t in tSet) <= (1 + w[1] + w[2])* gp.quicksum(r_coeff[t] * UC * u_mincost[t] for t in tSet)), name='UD_')
        gep.update()
        
        obj = w[0] * cost_coeff * (gp.quicksum( ((1+r)**(-t)) * (TIC[t] + TMC[t] + TPC[t] + TUC[t]+ TFTC[t] + TVTC[t] + TCC[t] - CS[t]) for t in tSet)-SV) + w[1] * CO2_coeff * gp.quicksum( GCE[t] - VCES[t] for t in tSet) + w[2] * NOX_coeff * gp.quicksum( GNE[t] - VNES[t] for t in tSet)
        gep.setObjective(obj, GRB.MINIMIZE)
        gep.optimize()
        
        results['SolTime'].append( round( (perf_counter() - strt)/60 , 1))
        results['iter'].append(iter_n)
        
        df_s, df_ce, df_v, df_y, df_u, df_slack_NC, df_NC_tr = get_results()    
        save_results( maxT, w, df_s, df_ce, df_v, df_y, df_u, df_slack_NC, df_NC_tr )
        
#%% main

# mixed obj weight combinations
objWeights_list = [[0.80,  0.20, 0.00],
                   [0.70,  0.30, 0.00],
                   [0.60,  0.40, 0.00],
                   [0.50,  0.50, 0.00],
                   [0.40,  0.60, 0.00],
                   [0.30,  0.70, 0.00],
                   [0.20,  0.80, 0.00] ]
'''
objWeights_list = [[0.80,  0.20, 0.00],
                   [0.70,  0.30, 0.00],
                   [0.60,  0.40, 0.00],
                   [0.50,  0.50, 0.00],
                   [0.40,  0.60, 0.00],
                   [0.30,  0.70, 0.00],
                   [0.20,  0.80, 0.00],
                   [0.80,  0.00, 0.20],
                   [0.70,  0.00, 0.30],
                   [0.60,  0.00, 0.40],
                   [0.50,  0.00, 0.50],
                   [0.40,  0.00, 0.60],
                   [0.30,  0.00, 0.70],
                   [0.20,  0.00, 0.80],
                   [0.80,  0.10, 0.10],
                   [0.70,  0.15, 0.15],
                   [0.60,  0.20, 0.20],
                   [0.50,  0.25, 0.25],
                   [0.40,  0.30, 0.30],
                   [0.30,  0.35, 0.35],
                   [0.20,  0.40, 0.40]]
'''
solve_model(iter_n=0)

# Save summary results dictionary to cwd
save_summary_results()

#%% end of code

#%% load results from all excel files in current working directory into one Excel File 
'''
run this in the folder that include the gep result excel files with different weights
'''

dfs, dfce, dfv, dfu, dfd, df_slack_NC, df_NC_tr = load_results()

all_results_f_name = 'GEP_Results_all.xlsx'
with pd.ExcelWriter(all_results_f_name) as writer:
    dfs.to_excel(writer, sheet_name="s", index=False)
    dfce.to_excel(writer, sheet_name="obj", index=False)
    dfv.to_excel(writer, sheet_name="v", index=False)
    dfu.to_excel(writer, sheet_name="u", index=False)
    dfd.to_excel(writer, sheet_name="d", index=False)
    df_slack_NC.to_excel(writer, sheet_name="slack", index=False)
    df_NC_tr.to_excel(writer, sheet_name="NC_tr", index=False)

results_df = pd.read_excel('results.xlsx').reset_index()

#%% Visual Graphs for Model Results that exist in the current working directory

# calculate CO2_coeff and NOX_coeffs
minWeights_list = [[1.00,  0.00, 0.00],
                   [0.00,  1.00, 0.00],
                   [0.00,  0.00, 1.00]]

minmax_results = {'Model':[], 'Cost':[], 'CO2':[], 'NOX':[]}
for w_ in minWeights_list:
    
    dfce_ = dfce[(dfce['cost'] == w_[0]) & (dfce['CO2'] == w_[1]) & (dfce['NOX'] == w_[2]) ]
    
    if w_[0] == 1:
        cost_with_minCost = dfce_['NPV'].sum()
        CO2_with_minCost = dfce_['NetCO2'].sum()
        NOX_with_minCost = dfce_['NetNOx'].sum()
        
    if w_[1] == 1:
        cost_with_minCO2 = dfce_['NPV'].sum()
        CO2_with_minCO2 = dfce_['NetCO2'].sum()
        NOX_with_minCO2 = dfce_['NetNOx'].sum()
        
    if w_[2] == 1:
        cost_with_minNOX = dfce_['NPV'].sum()
        CO2_with_minNOX = dfce_['NetCO2'].sum()
        NOX_with_minNOX = dfce_['NetNOx'].sum()
    
cost_coeff = 1

CO2_coeff = (cost_with_minCO2 - cost_with_minCost)/(CO2_with_minCost - CO2_with_minCO2) 
NOX_coeff = (cost_with_minNOX - cost_with_minCost)/(NOX_with_minCost - NOX_with_minNOX)

minObj1, maxObj1 = results_df['Cost'].min(), results_df['Cost'].max()
minObj2, maxObj2 = results_df['CO2'].min(), results_df['CO2'].max()
minObj3, maxObj3 = results_df['NOX'].min(), results_df['NOX'].max()

CO2_coeff = (maxObj1 - minObj1)/(maxObj2 - minObj2)
NOX_coeff = (maxObj1 - minObj1)/(maxObj3 - minObj3)

# filter-out some weight combinations that doesnt need to be included (optional)
dfs = dfs[dfs['NOX'] != 1]
dfce = dfce[dfce['NOX'] != 1]
dfv = dfv[dfv['NOX'] != 1]
dfu = dfu[dfu['NOX'] != 1]
dfd = dfd[dfd['NOX'] != 1]
df_slack_NC = df_slack_NC[df_slack_NC['NOX'] != 1]

# make a list of the obj Weight combinations 
txt_arr = dfce['w'].unique()
objW_toAnalyze = []
for txt in txt_arr:
    w_arr = txt.split(' ')
    w_arr = [float(ww)/100 for ww in w_arr]
    objW_toAnalyze.append(w_arr)
objW_toAnalyze.sort()

w_list = []
for w_ in objW_toAnalyze:
    #w_list.append(str(int(100*w_[0])) + '-' + str(int(100*w_[1])) + '-' + str(int(100*w_[2])))
    w_list.append(str(int(100*w_[0])) + '-' + str(int(100*w_[1])) )

# define the color palette for unit types  (for plots)
color_dict = {
    'coal': "peru",
    'naturalGas': "sandybrown",
    'bioMass': "forestgreen",        
    'geoThermal': "darkorange",      
    'hydroRiver': "deepskyblue",     
    'hydroDam': "steelblue",         
    'wind': "lightskyblue",
    'solar': "gold",
    'nuclear': "dimgray"
}

sub_colors = {i: color_dict[i] for i in color_dict if i in subK}

# calculate objVal and normalized ObjVal_
objVal = {}
objVal_ = {}

for w_ in objW_toAnalyze:
    w_id = str(int(100*w_[0])) + '-' + str(int(100*w_[1])) + '-' + str(int(100*w_[2]))
    
    dfce_ = dfce[(dfce['cost'] == w_[0]) & (dfce['CO2'] == w_[1]) & (dfce['NOX'] == w_[2]) ]

    # obj function value
    objVal[w_id] = w_[0] * dfce_['NPV'].sum() + w_[1] * CO2_coeff * dfce_['NetCO2'].sum() + w_[2] * NOX_coeff * dfce_['NetNOx'].sum()
    # obj function value (scaled to 0-1 interval)
    objVal_[w_id] = w_[0] * ( (dfce_['NPV'].sum() - minObj1)/(maxObj1 - minObj1) ) + w_[1] * (dfce_['NetCO2'].sum() - minObj2) / (maxObj2 - minObj2) + w_[2] * (dfce_['NetNOx'].sum() - minObj3) / (maxObj3 - minObj3)
    

# plot objVal of the weight combinations
fig = plt.subplots(figsize=(10,4))
plt.scatter(w_list, [objVal[w_] for w_ in objVal])
plt.ylabel("Obj Value")
plt.xticks(rotation = 90)
plt.show()

# plot normalized objVal_ of the weight combinations
fig = plt.subplots(figsize=(10,4))
plt.scatter(w_list, [objVal_[w_] for w_ in objVal_])
plt.ylabel("Obj Value_")
plt.xticks(rotation = 90)
plt.show()

#%%

# new capacity investments plot (total in planning period) 
# objweights on x-axis, GW in y-axis    
fig = plt.subplots(figsize=(10,4))
lb = [0]*len(objW_toAnalyze)
for k in subK:
    y = []
    for w_ in objW_toAnalyze:
        dfs_ = dfs[ (dfs['cost']==w_[0]) & (dfs['CO2']==w_[1]) & (dfs['NOX']==w_[2])]
        dfs_ = dfs_[ dfs_['k']==k ]
        if len(dfs_) > 0:
            y.append(dfs_['s'].sum())
        else:
            y.append(0)
    plt.bar(w_list, y, bottom=lb, color=color_dict[k])
    lb = [lb[w] + y[w] for w in range(len(objW_toAnalyze))]
plt.ylabel("GW")
plt.legend(subK, fontsize=12)
plt.xticks(rotation=90, fontsize=12)
plt.ylim(0,max(lb).round(0) + 1)
plt.title("Total New Capacity with Different Obj Weights", fontsize=12)
plt.show()

# new capacity investments plot 
max_s = int(dfs.groupby(['t','cost','CO2','NOX'])['s'].sum().max()) + 1
n_rows, n_cols = 3, 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 12), sharex=True, sharey=True)
axes = axes.flatten()
years = list(tSet)
plt.rcParams.update({'font.size': 10})
legend_handles, legend_labels = None, None

for w_ in objW_toAnalyze:
    ax = axes[objW_toAnalyze.index(w_)]
    
    # new capacity investments plots (year-by-year in planning period) 
    s_ = {(t,b,k):0 for t in tSet for b in B for k in K}
    dfs_ = dfs[(dfs['cost'] == w_[0]) & (dfs['CO2'] == w_[1]) & (dfs['NOX'] == w_[2]) ]    
    for i in dfs_.index:
        s_[(dfs_.loc[i]['t'], dfs_.loc[i]['b'], dfs_.loc[i]['k'])] = dfs_.loc[i]['s']
    
    lb = [0]*len(tSet)
    i = 0
    for k in subK:
        i += 1
        y = [np.sum([s_[(t,b,k)] for b in B]) for t in tSet]
        if i == 1 : l1 = ax.bar(years, y, bottom=lb, color=sub_colors[k])
        if i == 2 : l2 = ax.bar(years, y, bottom=lb, color=sub_colors[k])
        if i == 3 : l3 = ax.bar(years, y, bottom=lb, color=sub_colors[k])
        if i == 4 : l4 = ax.bar(years, y, bottom=lb, color=sub_colors[k])
        
        lb = [lb[t] + y[t] for t in tSet] 
        
    if objW_toAnalyze.index(w_) == 0:
        legend_handles = [l1, l2, l3, l4]
        legend_labels = subK

    ax.set_title(f"{w_list[objW_toAnalyze.index(w_)]}", fontsize=16)   
    ax.set_xlim(0, len(tSet))
    ax.set_ylim(0, max_s)
    ax.set_xticks(tSet)  # adjust if needed

fig.supylabel("GW")
#fig.suptitle("New Capacity Investments with Different Obj Weights")
fig.legend(legend_handles, legend_labels, loc="lower center", ncol=4, fontsize=16)

plt.tight_layout(rect=[0.02, 0.02, 0.95, 0.95], h_pad=1.2, w_pad=1.2)
for ax in axes:
    ax.label_outer()
plt.show()

# generation mix in the initial year
dfd_ = dfd[ dfd['t'] == tSet[-1] ]
peak_load = dfd_.groupby(['g','h'])['d'].sum().max()
gen = {}

for w_ in [ [1,0,0], [0.5, 0.5, 0], [0,1,0] ]:
    for tt in [0,tSet[-1]]:
        dfv_ = dfv[(dfv['cost'] == w_[0]) & (dfv['CO2'] == w_[1]) & (dfv['NOX'] == w_[2]) & (dfv['t'] == tSet[tt]) ]
        dfd_ = dfd[(dfd['cost'] == w_[0]) & (dfd['CO2'] == w_[1]) & (dfd['NOX'] == w_[2]) & (dfd['t'] == tSet[tt]) ]
        
        g_list = sorted(dfv_['g'].unique(), key=lambda x: int(x[1:]))
        h_list = sorted(dfv_['h'].unique(), key=lambda x: int(x[1:])) 
        v_ = {(g, h, k): 0 for g in g_list for h in h_list for k in K}
        for _, row in dfv_.iterrows():
            key = (row['g'], row['h'], row['k'])
            v_[key] = row['v']
        
        G_sorted = sorted(dfv_['g'].unique(), key=lambda x: int(x[1:]))
        H_sorted = sorted(dfv_['h'].unique(), key=lambda x: int(x[1:]))
        
        x_labels = [f"{g}" for g in G_sorted for h in H_sorted]
        x = np.arange(len(x_labels))  # X-axis locations
        bar_width = 0.9  
        lb = np.zeros(len(x))
        
        fig, ax = plt.subplots(figsize=(30, 12))
        for k in K:
            y = []
            for g in G_sorted:
                for h in H_sorted:
                    y.append(v_[(g, h, k)] / weight_g[G.index(g)])
            ax.bar(x, y, bottom=lb, width=bar_width, color=color_dict[k], label=k)
            lb += np.array(y)
            gen[(w_[0], tt, k)] = sum(y)
        
        weight_map = dict(zip(G, weight_g))
        dfd_['scaled_d'] = dfd_['d'] / dfd_['g'].map(weight_map)
        ax.plot(x, dfd_['scaled_d'])
        
        ax.set_xticks(x[::24])  # Show one tick per day (24 hours = one day)
        ax.set_xticklabels(G_sorted)
        ax.set_ylabel("GWh")
        ax.set_ylim(0,105)
        ax.set_title(f"Generation Mix at the End of Planning Period {w_list[objW_toAnalyze.index(w_)]} year-{tt}")
        ax.legend(title="Unit Type")
        plt.tight_layout()
        plt.show()

df = pd.DataFrame(
    [(keys[0], keys[1], keys[2], vals) for keys, vals in gen.items()],
    columns=["w_cost", "tt", "k", "x"] )
df.to_excel("gen_output.xlsx", index=False)

#%% scatter plots and pareto front for each pair of criteria 
df_ = dfce.groupby('w')[['NPV','NetCO2', 'NetNOx']].sum()

# Cost vs CO2
dominated_list = []
for w in df_.index:
    isDominated = False
    for w_ in df_.index:
        if w != w_:
            if (df_.loc[w]['NPV'] > df_.loc[w_]['NPV']) and (df_.loc[w]['NetCO2'] > df_.loc[w_]['NetCO2']):
                df_.loc[w]['dominated'] = True
                isDominated = True
    dominated_list.append(isDominated)
df_['isDominated_cost_co2'] = dominated_list

fig = plt.subplots(figsize=(8,8))
plt.scatter(df_['NPV'], df_['NetCO2'])
df_nonDominated = df_[df_['isDominated_cost_co2'] == False]
df_nonDominated = df_nonDominated.sort_values(['NPV', 'NetCO2'])
plt.plot(df_nonDominated['NPV'], df_nonDominated['NetCO2'], '--o')
plt.xlabel('Cost (Sum of NPV)', fontsize = 16)
plt.ylabel('Net CO2 Emission', fontsize = 16)


df_['Cost_'] = (df_['NPV'] - df_['NPV'].min()) / (df_['NPV'].max() - df_['NPV'].min())
df_['CO2_'] = (df_['NetCO2'] - df_['NetCO2'].min()) / (df_['NetCO2'].max() - df_['NetCO2'].min())

fig = plt.subplots(figsize=(8,8))
plt.scatter(df_['Cost_'], df_['CO2_'])
df_nonDominated = df_[df_['isDominated_cost_co2'] == False]
df_nonDominated = df_nonDominated.sort_values(['Cost_', 'CO2_'])
plt.plot(df_nonDominated['Cost_'], df_nonDominated['CO2_'], '--o')
plt.xlabel('Min-Max Scaled Cost', fontsize = 16)
plt.ylabel('Min-Max Scaled CO2 Emission', fontsize = 16)

#%%
# Cost vs NOX
df_['NOX_'] = (df_['NetNOX'] - df_['NetNOX'].min()) / (df_['NetNOX'].max() - df_['NetNOX'].min())

dominated_list = []
for w in df_.index:
    isDominated = False
    for w_ in df_.index:
        if w != w_:
            if (df_.loc[w]['NPV'] > df_.loc[w_]['NPV']) and (df_.loc[w]['NetNOx'] > df_.loc[w_]['NetNOx']):
                df_.loc[w]['dominated'] = True
                isDominated = True
    dominated_list.append(isDominated)
df_['isDominated_cost_nox'] = dominated_list

fig = plt.subplots(figsize=(8,8))
plt.scatter(df_['NPV'], df_['NetNOx'])
df_nonDominated = df_[df_['isDominated_cost_nox'] == False]
df_nonDominated = df_nonDominated.sort_values(['NPV', 'NetNOx'])
plt.plot(df_nonDominated['NPV'], df_nonDominated['NetNOx'], '--o')
plt.xlabel('Cost (Sum of NPV)', fontsize = 16)
plt.ylabel('Net NOX Emission', fontsize = 16)

fig = plt.subplots(figsize=(8,8))
plt.scatter(df_['Cost_'], df_['NOX_'])
df_nonDominated = df_[df_['isDominated_cost_nox'] == False]
df_nonDominated = df_nonDominated.sort_values(['Cost_', 'NOX_'])
plt.plot(df_nonDominated['Cost_'], df_nonDominated['NOX_'], '--o')
plt.xlabel('Min-Max Scaled Cost', fontsize = 16)
plt.ylabel('Min-MAx Scaled NOX Emission', fontsize = 16)

# CO2 vs NOX
dominated_list = []
for w in df_.index:
    isDominated = False
    for w_ in df_.index:
        if w != w_:
            if (df_.loc[w]['NetCO2'] > df_.loc[w_]['NetCO2']) and (df_.loc[w]['NetNOx'] > df_.loc[w_]['NetNOx']):
                df_.loc[w]['dominated'] = True
                isDominated = True
    dominated_list.append(isDominated)
df_['isDominated_co2_nox'] = dominated_list

fig = plt.subplots(figsize=(8,8))
plt.scatter(df_['NetCO2'], df_['NetNOx'])
df_nonDominated = df_[df_['isDominated_co2_nox'] == False]
df_nonDominated = df_nonDominated.sort_values(['NetCO2', 'NetNOx'])
plt.plot(df_nonDominated['NetCO2'], df_nonDominated['NetNOx'], '--o')
plt.xlabel('Net CO2 Emission', fontsize = 16)
plt.ylabel('Net NOX Emission', fontsize = 16)

#df_.to_excel('pareto.xlsx', sheet_name="pareto")

#%% demand

bL = 1e-3 * baseLoad.sum(axis=1)
weighted_d_low = (weight_tj[:,None,None,:,None]* ( baseLoad + ((nEVs[:,None,None,None,None] * regional_EVLoad_)*(1/theta))) ).sum(axis=(3,4))
weighted_d_high = (weight_tj[:,None,None,:,None]* ( baseLoad + ((nEVs[:,None,None,None,None] * regional_EVLoad_)*(1/theta))) ).sum(axis=(3,4))

n_rows, n_cols = 4, 6
fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 16), sharex=True, sharey=True)
axes = axes.flatten()
hours = list(range(len(H)))
plt.rcParams.update({'font.size': 10})
legend_handles, legend_labels = None, None

for g in range(len(subG)):
    ax = axes[g]
    l1, = ax.plot(hours, [bL[tSet[0], g, h] for h in range(len(H))], "--")
    l2, = ax.plot(hours, [bL[tSet[-1], g, h] for h in range(len(H))], "--")
    l3, = ax.plot(hours, [weighted_d_low[tSet[-1], g, h] for h in range(len(H))], "-")
    l4, = ax.plot(hours, [weighted_d_high[tSet[-1], g, h] for h in range(len(H))], "-")

    if g == 0:
        legend_handles = [l1, l2, l3, l4]
        legend_labels = [
            f"BaseL t={tSet[0]}",
            f"BaseL t={tSet[-1]}",
            f"BaseL + EVLoad_Low t={tSet[-1]}",
            f"BaseL + EVLoad_High t={tSet[-1]}"]
    
    if g%2 == 0:
        g_type= f"month-{int(g/2 + 1)} workday"
    else:
        g_type= f"month-{int((g-1)/2 + 1)} nonworkday"

    ax.set_title(f"{g_type}")   
    ax.set_xlim(0, len(H)-1)
    ax.set_ylim(25, 105)
    ax.set_xticks([0, 6, 12, 18, 23])  # adjust if needed

# If there are exactly 24 panels, this does nothing; otherwise hide extras
for k in range(len(subG), n_rows*n_cols):
    fig.delaxes(axes[k])

# Shared axis labels and global legend
fig.supxlabel("Hour of Day")
fig.supylabel("GW")
fig.legend(legend_handles, legend_labels, loc="upper center", ncol=4, fontsize=12)

plt.tight_layout(rect=[0.02, 0.02, 0.95, 0.95], h_pad=1.2, w_pad=1.2)
for ax in axes:
    ax.label_outer()
plt.show()

#%%
# Demand (baseload + EV load) _________________________________________________

# baseLoad per (g, h): hourly loads per each representative day g
df = pd.read_excel(source_f,sheet_name='baseLoad(g)')
df = df[df['g'].isin(subG)]
temp = {(df.loc[i]['g'],hr): df.loc[i][hr] for hr in H for i in df.index}
baseLoad_ = np.zeros((len(subG), len(H)))
for g_idx, g in enumerate(df['g'].unique()):
    for hr in H:
        baseLoad_[g_idx, H.index(hr)] = temp[(g, hr)]
del temp

# baseLoad weights per representative day g (weights sum up to 365)
df = pd.read_excel(source_f,sheet_name='baseLoad(g)')
if len(subG) == len(G):
    weight_g = df['weight'].to_numpy()
else:
    # if only a subset of repdays are used then rearrange the weights
    unselected_weight = df[~df['g'].isin(subG)]['weight'].sum()
    selected_weight = df[df['g'].isin(subG)]['weight'].sum()
    df = df[df['g'].isin(subG)]
    weight_g = df['weight'].to_numpy()
    for g in range(len(weight_g)):
        weight_g[g] = weight_g[g] + unselected_weight * (weight_g[g] / selected_weight)

# baseLoad weights per (g,b): consumption share of each region b on representative day g 
# (for each day g weights of regions sum up to 1)
df = pd.read_excel(source_f,sheet_name='baseLoadWeight(g,b)')
df = df[df['g'].isin(subG)].set_index('g')
baseload_weight_g_b = df.to_numpy()[:,:len(B)] 

# baseLoad per (t,b,g,h) (each year t's baseload is icreased by rBase rate)
baseLoad = np.zeros((len(tSet), len(B), len(subG), len(H)))
for t in range(len(tSet)):
    for b in range(len(B)):
        for g in range(len(subG)):
            for h in range(len(H)):
                baseLoad[t,b,g,h] = ((1+rBase)**t) * baseload_weight_g_b[g,b] * baseLoad_[g,h]


# EV Load (g,h,j): hourly EV charging loads on representative day g with charger availability scenario (environment setting) j
df = pd.read_excel(source_f,sheet_name='10 000 EVLoad(g,j)')
df = df[df['g'].isin(subG)]
temp = {(df.loc[i]['g'], df.loc[i]['j'], hr): df.loc[i][hr] for hr in H for i in df.index}
EVLoad_ = np.zeros((len(subG), len(J), len(H) ))
for g in subG:
    for j in J:
        for hr in H:
            EVLoad_[subG.index(g), J.index(j), H.index(hr)] = temp[(g, j, hr)]
del temp

# Calculate the Average Driving Distance per EV (DD) based on representative EV Loads
'''
Explanation of DD calculation
365 : days
1000 * (1/cre) : km per MWh of energy consumption
np.sum([...])/10000 : Total daily EV load devided by the nEVs (10 000)
'''
cre_MWh_per_km = cre/1000  # conversion from kWh/km to MWh/km
DD_percar_gj = (EVLoad_.sum(axis=2) / cre_MWh_per_km) / 10000
DD = 0
for g_idx, g in enumerate(subG):
    for j_idx, j in enumerate(J):
        DD += weight_g[g_idx] * (1/len(J)) *  DD_percar_gj[g_idx, j_idx]
DD = round(DD,0)
   
# EVLoad weights per (g,b): EV Load share of each region b on representative day g 
# (for each day g weights of regions sum up to 1)
df = pd.read_excel(source_f,sheet_name='EVLoadWeight(g,b)').set_index('g')
EVLoad_weight_g_b = df.to_numpy()[:len(subG),:]

# EVLoad[t,b,g,j,h]: EV charging load weighted by g,b and scaled to nEVs[t]
EVLoad = np.zeros((len(tSet), len(B), len(subG), len(J), len(H)))
for t in range(len(tSet)):
    for b in range(len(B)):
        for g in range(len(subG)):
            for j in range(len(J)):
                for h in range(len(H)):
                    EVLoad[t,b,g,j,h] = EVLoad_[g,j,h] * (nEVs[t]/theta) * EVLoad_weight_g_b[g,b]

# Total Demand (baseload + EVLoad)
# d[t,g,h,j,b] : electricity demand of region b in year t day g hour h with scenario j 
d = np.zeros((len(tSet), len(subG), len(H), len(J), len(B)))
for t in range(len(tSet)):
    for g in range(len(subG)):
        for h in range(len(H)):
            for j in range(len(J)):
                for b in range(len(B)):
                    d[t,g,h,j,b] = 1e-3 * (baseLoad[t,b,g,h] + EVLoad[t,b,g,j,h])
#d = d.round(3)
    
# alpha[t,g,j] : weight of cha setting j on day g in year t
alpha = np.zeros((len(tSet), len(subG), len(J)))
for t in range(len(tSet)):
    for g in range(len(subG)):
        for j in range(len(J)):
            alpha[t,g,j] = weight_g[g] * weight_tj[t,j]    
            
# Delete Unneccesary Data that will not be used from now on...
#del baseLoad, baseLoad_, EVLoad, EVLoad_

