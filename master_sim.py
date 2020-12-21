import numpy as np
import pandas as pd
import pickle
import tracemalloc
import itertools
import sys, os
import datetime
from scipy import stats
from scipy import interpolate
import matplotlib.pyplot as plt
import statsmodels.api as sm
import xlsxwriter

sys.path.append("/Users/antoniaaguilera/Documents/GitHub/child-care-welfare")

#call py scripts
import utility as util
import parameters as parameters
import simdata as simdata
import estimate as est
import bootstrap as bstr


begin_time = datetime.datetime.now()
#------------ PREP DATA ------------#
data = pd.read_stata('/Users/antoniaaguilera/Documents/GitHub/child-care-welfare/data/data_python.dta')
N = len(data)
data['constant']=np.ones((N,1))

#------------ REG WITH DATA ------------#
reg=sm.OLS(endog=data['ln_w'], exog=data[['constant', 'm_sch']], missing='drop').fit()

#betas=[beta0, beta1]
betas = [reg.params[0], reg.params[1]]
sigmaw= np.var(reg.resid)

#------------ PARAMETERS ------------#

meanshocks = [0,0]
covshocks  = [[0.5,0],[0,0.5]] 
T          = (24-8)*20  #monthly waking hours
Lc         = 8*20       #monthly cc hours
alpha      = 0.1
gamma      = 0.1

#------------ VARIABLES ------------#

H1 = np.array(data[['monthly_hrs']], dtype=np.float64)
D1 = np.array(data[['d_cc_34']], dtype=np.float64)

#------------ CALL CLASSES ------------#
param0 = parameters.Parameters(betas,sigmaw, meanshocks, covshocks, T, Lc, alpha, gamma)

model     = util.Utility(N, data, param0)
model_sim = simdata.SimData(N, model)
model_est = est.estimate(N, data, param0, model, model_sim)

#------------ ESTIMATION SIM & BOOTSTRAP ------------#
times = 20
results_estimate = model_est.simulation(times)
results_bootstrap = bstr.bootstrap(data, times)

#------------ DATA2EXCEL ------------#
workbook  = xlsxwriter.Workbook('/Users/antoniaaguilera/Documents/GitHub/child-care-welfare/data/labor_choice.xlsx')
worksheet = workbook.add_worksheet()

worksheet.write('A1', 'nº of sims')
worksheet.write('B1', times )

worksheet.write('B2', 'parameter')
worksheet.write('B3', 'labor choice')
worksheet.write('B4', 'cc choice')
worksheet.write('B5', 'beta_0')
worksheet.write('B6', 'beta_1')
worksheet.write('B7', 'sigma^2_{varepsilon}')

worksheet.write('C2', 'sim')
worksheet.write('C3', results_estimate['Labor Choice'])
worksheet.write('C4', results_estimate['CC Choice'])
worksheet.write('C5', results_estimate['Beta0'])
worksheet.write('C6', results_estimate['Beta1'])
worksheet.write('C7', results_estimate['Resid var'])

worksheet.write('D2', 'data')
worksheet.write('D3', results_bootstrap['Labor Choice'])
worksheet.write('D4', results_bootstrap['CC Choice'])
worksheet.write('D5', results_bootstrap['Beta0'])
worksheet.write('D6', results_bootstrap['Beta1'])
worksheet.write('D7', results_bootstrap['Resid var'])

worksheet.write('E2', 'SE')
worksheet.write('E3', results_bootstrap['SE Labor Choice'])
worksheet.write('E4', results_bootstrap['SE CC Choice'])
worksheet.write('E5', results_bootstrap['SE Beta0'])
worksheet.write('E6', results_bootstrap['SE Beta1'])


        
workbook.close()
#------------ END TIME ------------#
end_time = datetime.datetime.now()
print(end_time-begin_time)
