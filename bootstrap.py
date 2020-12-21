#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 21:56:27 2020

@author: akac
"""
import numpy as np
import statsmodels.api as sm


def bootstrap(data, times):
        n = data.shape[0]
        smpl_mean_labor = np.zeros(times)
        smpl_mean_cc    = np.zeros(times)
        beta0_est = np.zeros(times)
        beta1_est = np.zeros(times)
        se0_est = np.zeros(times)
        se1_est = np.zeros(times)
        varres_est = np.zeros(times)
        
        for i in range(0,times):
            smpl = data.sample(n, replace=True)
            smpl_mean_labor[i] = np.mean(smpl['monthly_hrs']/160)
            smpl_mean_cc[i]    = np.mean(smpl['d_cc_34'])
            
            reg=sm.OLS(endog=smpl['ln_w'], exog=smpl[['constant', 'm_sch']], missing='drop').fit()
            
            beta0_est[i] = reg.params[0]
            beta1_est[i] = reg.params[1]
            se0_est[i]   = reg.bse[0] 
            se1_est[i]   = reg.bse[1]
            
            varres_est[i] = np.var(reg.resid)
            
        
        est_boot_labor = np.mean(smpl_mean_labor)
        est_boot_cc    = np.mean(smpl_mean_cc)
        
        beta0_boot = np.mean(beta0_est)
        beta1_boot = np.mean(beta1_est)
        
        se0_boot = np.mean(se0_est)
        se1_boot = np.mean(se1_est)
        varres   = np.mean(varres_est) 
    
        se_labor = np.std(smpl_mean_labor)/np.sqrt(n)
        se_cc    = np.std(smpl_mean_cc)/np.sqrt(n)


        return {'Labor Choice': est_boot_labor,
                'SE Labor Choice': se_labor, 
                'CC Choice': est_boot_cc,
                'SE CC Choice': se_cc,
                'Beta0': beta0_boot,
                'SE Beta0': se0_boot,
                'Beta1': beta1_boot,
                'SE Beta1': se1_boot,
                'Resid var': varres}
    