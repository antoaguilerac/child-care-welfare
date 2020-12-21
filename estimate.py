#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 13:47:22 2020

@author: akac
"""

import numpy as np
import pandas as pd
import pickle
import tracemalloc
import itertools
import sys, os
import statsmodels.api as sm
from scipy import stats
from scipy import interpolate
import matplotlib.pyplot as plt
import seaborn as sn
from statsmodels.iolib.summary2 import summary_col




class estimate:
    "This class "
    
    def __init__(self, N, data, param, model, model_sim):
        "Initial class"
        
        self.N, self.data, self.param = N, data, param
        self.model, self.model_sim    = model, model_sim
        
    
    def simulation(self,times):
        "Function that simulates x times."
        
        est_b0     = np.zeros(times)
        est_b1     = np.zeros(times)
        est_varres = np.zeros(times)
        mean_labor = np.zeros(times)
        mean_cc    = np.zeros(times)
        
        
        for i in range(0,times):
            
            sim = self.model_sim.choice()
            
            ln_wage = np.log(sim['Wage'])
            
            y = pd.DataFrame(ln_wage)
            
            reg = sm.OLS(endog=y, exog=self.data[['constant', 'm_sch']], missing='drop').fit()
            
            est_b0[i]  = reg.params[0]
            est_b1[i]  = reg.params[1]
            est_varres[i] = np.var(reg.resid)
            
            mean_labor[i] = np.nanmean(sim['Hours']/160) 
            mean_cc[i]    = np.nanmean(sim['CC'])
        
        sim_b0 = np.mean(est_b0)
        sim_b1 = np.mean(est_b1) 
        sim_varres = np.mean(est_varres)
        
        sim_labor = np.mean(mean_labor)
        sim_cc    = np.mean(mean_cc)
        

        plt.hist(est_b1, bins=100)
        plt.title("Histogram B1")
        plt.show()
        
        return { 'Labor Choice': sim_labor,
                'CC Choice': sim_cc,
                'Beta0': sim_b0,
                'Beta1': sim_b1,
                'Resid var': sim_varres}
    
        









