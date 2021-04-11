# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 23:21:37 2020

@author: Pilar Tagliero
"""

# Go to Tools >> Preferences >> IPython console >> Graphics >> Backend:Inline, change "Inline" to "Automatic", click "OK"
# Reset the kernel at the console, and the plot will appear in a separate window

# To the source code of a function:
# import inspect 
# print(inspect.getsource(control.hinfsyn))

import numpy as np
from functions import airfoil, V_flutter, value2norm, norm2value, objective, constraints, mass_velocities, plot_history, save_results
from AE_model import AE_model_OL, AE_model_CL
from math import sqrt
import control as ctrl
import math
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from time import perf_counter 
import os

### LOAD DATA
#Goland wing model
s = 6.096 #[m] =20ft semispan of the wing 
c = 1.829 #[m] =6ft chord of the section 
rho = 38.2723 #density of the material
E = 1.2062e8 #Young's modulus 
G = 8.044e6 #Shear modulus
h = 0.5 #[m] =0.5 height of the section 
ea_b = 0.3 #elastic axis position from the leading edge normalized with the semichord (b)

# Aerodynamic coefficients
cl_alpha = 6.28 #[1/rad]
cm_alpha = 0 #symmetrical airfoil 

rho_air = 1.225 #[kg/m3]

# Ratio of the flap chord with respect to the total chord
EF = 0.2

# Thicknesses

thf = (0.25*c)/10 #10 or 12

th1 = c*(1-EF)/10
th2 = c*(1-EF)/10 
th3 = h/10
th4 = h/10
th5 = c*(1-EF)/10


th = np.array([th1, th2, th3, th4, th5]) #thickness of the cross section

# ORGANIZE DATA
mat_prop = {'rho':rho, 'E':E, 'G':G}
geom_prop = {'s':s, 'c':c, 'ea_b':ea_b, 'EF':EF, 'h':h, 'th':th, 'thf': thf}
aero_coef = {'cl_alpha':cl_alpha, 'cm_alpha':cm_alpha}

         
#    ################# th4 ########
#    #             #              #
#    #             #              #
#   th1           th5            th2
#    #             #              #
#    #             #              #
#    ################# th3 ########


#%% PLANT: WING MODEL

[m_init, I_alpha, w_h, w_alpha, xg, J, x_alpha] = airfoil (geom_prop, mat_prop)

# Flight velocities considered
V = np.arange(1, 250.5, 0.1).tolist()

# Output of the state space representation
output_choice = np.array([1, 1, 1, 1])

# State space representation
wing = AE_model_OL(aero_coef, rho_air, ea_b, c, s, m_init, I_alpha, x_alpha, w_alpha, w_h, output_choice, EF)

[Vf_OL_init, wn_f_OL]  = V_flutter(wing, V, output_choice)


#%% CONTROLLER

# d_max = RL/4f, f:max frequency at flutter velocity
# d_max = 0.5*Kh*d_max 
# d_max = 0.5*Ka*alpha_max
# d_max/(f/2*pi) = 0.5*Khd*d_max 
# d_max/(f/2*pi) = 0.5*Kad*alpha_max

RL = 150
h_max = 1
alpha_max = 20

d_max = RL/(4*wn_f_OL[1]/(2*math.pi))

# Boundaries
Kh = math.radians(0.5*d_max/h_max)
Ka = 0.5*d_max/alpha_max

Khd = math.radians(0.5*d_max/(h_max*wn_f_OL[1]))
Kad = 0.5*d_max/(alpha_max*wn_f_OL[1])


#%%Common for all minimizations

# Initialization values
K = np.array([-0.01, 0.05, -0.002, 0.0003])
# K = np.array([-0.01, 0.05, -0.001, 0.0003])

## Initialization
DV_init = np.concatenate((th, K, [EF])) #Design variables

lb_DV = np.array([geom_prop['c']*(1-0.1)/20, geom_prop['c']*(1-0.1)/20, geom_prop['h']/20, geom_prop['h']/20, 1e-6, -Kh, -Ka, -Khd, -Kad, 0.1])
ub_DV = np.array([geom_prop['c']*(1-0.1)/10, geom_prop['c']*(1-0.1)/10, geom_prop['h']/10, geom_prop['h']/10, geom_prop['c']*(1-0.1)/10, Kh, Ka, Khd, Kad, 0.25])

x_init = value2norm(DV_init, lb_DV, ub_DV)
lb = value2norm(lb_DV, lb_DV, ub_DV)
ub = value2norm(ub_DV, lb_DV, ub_DV)

## Reference values
Ref = np.array([m_init, Vf_OL_init])

# Constraints
myConstr = ({'type': 'ineq', 'fun': constraints, 'args':(rho_air, geom_prop, mat_prop, Ref, lb_DV, ub_DV, V, output_choice, aero_coef)})


# Save data at each step of the optimization
xk = [x_init]

#Callback
def callbackFunc(xi):
    global xk        
    xk.append(xi)

# Boundaries for the optimization
Bnd = Bounds(lb, ub, keep_feasible=True) #min, max

# Optimization method
Met = 'SLSQP' # Sequential Least SQuares Programming (SLSQP) Algorithm
#Met = 'COBYLA'
#Met = 'trust-constr'

#%% OPTIMIZATION - MINIMIZE - this one works

time_start = perf_counter()
tol = 1e-4
f = 2
eps = 0.05

# Optimization
opt = minimize(objective, 
               x_init, 
               args=(rho_air, geom_prop, mat_prop, Ref, lb_DV, ub_DV, V, output_choice, aero_coef, f,),
               method=Met,
               jac = '2-point',
               bounds=Bnd,
               constraints=myConstr,
               tol = tol,
               options={'disp': True, 'eps':eps},
               callback=callbackFunc)

time_end = perf_counter()

time_elapsed = (time_end - time_start)

print("Elapsed time during the whole program in seconds:", time_elapsed) 

#%% ORGANIZE AND PLOT RESULTS 

#xk.append(opt.x)

k = np.arange(0, len(xk), 1)
    
xk_value = []
    
for it in k:
    x = norm2value(xk[it], lb_DV, ub_DV)
    xk_value.append(x)
 
history = {'th1':[], 'th2':[], 'th3':[], 'th4':[], 'th5':[], 'Ka':[], 'Kad':[], 'Kh':[], 'Khd':[], 'm':[], 'Vf_OL':[], 'Vf_CL':[], 'f_obj':[], 'EF':[]}

for it in k:
    history['th1'].append(xk_value[it][0])
    history['th2'].append(xk_value[it][1])
    history['th3'].append(xk_value[it][2])
    history['th4'].append(xk_value[it][3])
    history['th5'].append(xk_value[it][4])
            
    history['Kh'].append(xk_value[it][5])
    history['Ka'].append(xk_value[it][6])
    history['Khd'].append(xk_value[it][7])
    history['Kad'].append(xk_value[it][8])
    
    history['EF'].append(xk_value[it][9])
        
## Mass and flutter velocities for each xk
    [m, Vf_OL, Vf_CL] = mass_velocities(it, aero_coef, history, rho_air, geom_prop, mat_prop, V, output_choice)
        
    history['m'].append(m)
    history['Vf_OL'].append(Vf_OL)
    history['Vf_CL'].append(Vf_CL)
    
    if f==1:
        history['f_obj'].append(history['m'][it]/Ref[0] - history['Vf_CL'][it]/Ref[1])
    else:
        history['f_obj'].append(history['m'][it]/Ref[0])

#Plot and save results
case = 'f'+str(f)+'eps'+str(eps)
os.mkdir('Results/'+str(case))
plot_history (range(0,len(xk)), history, case)
save_results(case, history, opt, time_elapsed)
plt.close('all')

##%% OPTIMAL DESIGN 
#
#DV = np.array([ 0.85280246, -1.        ,  0.94975514,  0.94975514, -0.46666422,
#        0.29754518,  1.        ,  1.        ,  0.84215599,  0.39679175])
#DV = opt.x
#
#DV = norm2value(DV, lb_DV, ub_DV)
#geom_prop['th'] = DV[0:5]
#K = DV[5:9]
#EF = DV[9]
#[m, I_alpha, w_h, w_alpha, xg, J, x_alpha] = airfoil (geom_prop, mat_prop)
#wing_CL = AE_model_CL(aero_coef, rho_air, geom_prop['ea_b'], geom_prop['c'], geom_prop['s'], m, I_alpha, x_alpha, w_alpha, w_h, output_choice, EF, K)
#wing_OL = AE_model_OL(aero_coef, rho_air, geom_prop['ea_b'], geom_prop['c'], geom_prop['s'], m, I_alpha, x_alpha, w_alpha, w_h, output_choice, EF)
#
#### Closed-loop Analysis
#plt.grid()
#plt.title('Pole migration for the optimal design in OL and CL', fontsize=16, wrap=True)
#V = np.arange(1, 150, 2.5).tolist()
#for i in range(len(V)):
#    wing_CL.gen_SS(V[i],0,output_choice)
#    _, _, polesCL = ctrl.damp(wing_CL.SS, doprint=False)
#    plt.plot(np.real(polesCL),np.imag(polesCL),  linestyle = 'None', color='#000000', marker = 'o', markersize=5)
#    plt.xlabel('Real', fontsize=16)
#    plt.ylabel('Imag', fontsize=16)
#for i in range(len(V)):
#    wing_OL.gen_SS(V[i],0,output_choice)
#    _, _, polesOL = ctrl.damp(wing_OL.SS, doprint=False)
#    plt.plot(np.real(polesOL),np.imag(polesOL),  linestyle = 'None', color='#929591', marker = '^', markersize=5)
#plt.plot(np.real(polesCL),np.imag(polesCL),  linestyle = 'None', color='#000000', marker = 'o', markersize=5, label='Closed-Loop')
#plt.plot(np.real(polesOL),np.imag(polesOL),  linestyle = 'None', color='#929591', marker = '^', markersize=5, label='Open Loop')
#plt.legend(loc='best', fontsize=16)
#
#
##%% Robustness of OPTIMAL DESIGN 
#
#DV = norm2value(opt.x, lb_DV, ub_DV)
##Optimal
#geom_prop['th'] = DV[0:5]
#K = DV[5:9]
#EF = DV[9]
#[m, I_alpha, w_h, w_alpha, xg, J, x_alpha] = airfoil (geom_prop, mat_prop)
#wing_CL = AE_model_CL(aero_coef, rho_air, geom_prop['ea_b'], geom_prop['c'], geom_prop['s'], m, I_alpha, x_alpha, w_alpha, w_h, output_choice, EF, K)
#wing_CL.gen_SS(history['Vf_OL'][-1], 0, output_choice)
#X0 = np.array([1, 1, 0, 0])
#T = np.arange(0, 60, 1e-3)
#t, y_o = ctrl.initial_response(wing_CL.SS, T=T, X0=X0)
#
## 0.97 of Optimal
#geom_prop['th'] = 0.96*DV[0:5]
#K = DV[5:9]
#EF = DV[9]
#[m, I_alpha, w_h, w_alpha, xg, J, x_alpha] = airfoil (geom_prop, mat_prop)
#wing_CL = AE_model_CL(aero_coef, rho_air, geom_prop['ea_b'], geom_prop['c'], geom_prop['s'], m, I_alpha, x_alpha, w_alpha, w_h, output_choice, EF, K)
#wing_CL.gen_SS(history['Vf_OL'][-1], 0, output_choice)
#t, y_no = ctrl.initial_response(wing_CL.SS, T=T, X0=X0)
#
#fig, axs = plt.subplots(2,2)
#
#axs[0, 0].plot(t, y_o[0], linestyle='-', color='#000000', linewidth=1)
#axs[0, 0].set_title('$h$', fontsize=16)  
#axs[0, 0].set_ylabel('Optimal Design', fontsize=16)  
#axs[1, 0].plot(t, y_no[0], linestyle='-', color='#000000', linewidth=1, label='0.96% Optimal Design')
#axs[0, 0].grid(True)
#axs[1, 0].grid(True)
#axs[0, 1].set(xlim=[0,2]) 
#axs[0, 0].set(xlim=[0,2]) 
#axs[0, 1].plot(t, y_o[1], linestyle='-', color='#000000', linewidth=1)
#axs[0, 1].set_title(r'$\alpha$', fontsize=16)  
#axs[1, 0].set_ylabel ('96% Optimal Design', fontsize=16)   
#axs[1, 1].plot(t, y_no[1], linestyle='-', color='#000000', linewidth=1)    
#axs[1, 1].set(xlim=[0,60]) 
#axs[1, 0].set(xlim=[0,60]) 
#axs[0, 1].grid(True)    
#axs[1, 1].grid(True)   
#fig.suptitle('Simulation at $V_{F}^{OL}$', fontsize=16)
#plt.show()

