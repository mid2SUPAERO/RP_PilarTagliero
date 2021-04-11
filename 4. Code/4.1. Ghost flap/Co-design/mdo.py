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
from AE_model import AE_model_OL 
import math
import control as ctrl
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from time import perf_counter 
import os


### LOAD DATA
#Goland wing model
s = 6.096 #[m] =20ft semispan of the wing 
c = 1.829 #[m] =6ft chord of the section 
h = 0.5 #[m] height of the section

rho = 38.2723 #density of the material
E = 1.2062e8 #Young's modulus 
G = 8.044e6 #Shear modulus

th1 = c/15
th2 = c/15
th3 = h/15
th4 = h/15
th5 = c-2*c/15

#    ################# th4 ########
#    #             #              #
#    #             #              #
#   th1           th5            th2
#    #             #              #
#    #             #              #
#    ################# th3 ########

ea_b = 0.3 #elastic axis position from the leading edge normalized with the semichord (b)

# Aerodynamic coefficients for steady aerodynamics
cl_alpha = 6.28 #[1/rad]
cm_alpha = 0 #symmetrical airfoil 

rho_air = 1.225 #[kg/m3]

#ratio of the flap chord with respect to the total chord
EF = 0.2

# ORGANIZE DATA
th = np.array([th1, th2, th3, th4, th5]) #thickness of the cross section
mat_prop = {'rho':rho, 'E':E, 'G':G}
geom_prop = {'s':s, 'c':c, 'h':h, 'th':th, 'ea_b':ea_b}
aero_coef = {'cl_alpha':cl_alpha, 'cm_alpha':cm_alpha}

#%% PLANT: WING MODEL

[m_init, I_alpha, w_h, w_alpha, xg, J, x_alpha] = airfoil (geom_prop, mat_prop)

# Flight velocities considered
V = np.arange(1, 250.5, 0.1).tolist()

# Output of the state space representation
output_choice = np.array([1, 1, 1, 1])

# State space representation
wing = AE_model_OL(aero_coef, rho_air, ea_b, c, s, m_init, I_alpha, x_alpha, w_alpha, w_h, output_choice, EF)


[Vf_OL_init, wn_f_OL, SS_f, poles_f] = V_flutter(wing, V, output_choice)

#X0 = np.array([0.1, 0.1, 0, 0])
#T = np.arange(0, 10, 1e-3)
#t, y = ctrl.initial_response(SS_f, T=T, X0=X0)
#plt.plot(t, y[1], linestyle='-', color='y', linewidth=1, label='alpha')
#plt.plot(t, y[0], linestyle='-', color='r', linewidth=1, label='h')
#plt.xlabel('Time (s)')
#plt.title('States simulation at flutter condition')
#plt.legend(loc='best')
#plt.grid(True)
#plt.xlim([0, 5])
#plt.show()
#
# Open-loop Analysis
#plt.grid()
#plt.title('Pole migration when increasing flight speed',fontsize=16)
#V = np.arange(1, 170.5, 2.5).tolist()
#for i in range(len(V)):
#    wing.gen_SS(V[i],0,output_choice)
#    _, _, poles = ctrl.damp(wing.SS, doprint=False)
#    poles1 = [poles[0], poles[1]]
#    poles2 = [poles[2], poles[3]]
#    plt.plot(np.real(poles1),np.imag(poles1),  linestyle = 'None', color='#000000', marker = 'o', markersize=5)
#    plt.plot(np.real(poles2),np.imag(poles2),  linestyle = 'None', color='#929591', marker = '^', markersize=5)
#    plt.xlabel('Real', fontsize=16)
#    plt.ylabel('Imag', fontsize=16)
#    plt.pause(2e-5)
#plt.plot(np.real(poles1),np.imag(poles1),  linestyle = 'None', color='#000000', marker = 'o', markersize=5, label='Pitch Mode')
#plt.plot(np.real(poles2),np.imag(poles2),  linestyle = 'None', color='#929591', marker = '^', markersize=5, label='Plunge Mode')
#plt.legend(loc='best', fontsize=16)

# Open-loop Analysis
#plt.grid()
#V = np.arange(1, 170.5, 2.5).tolist()
#for i in range(len(V)):
#    wing.gen_SS(V[i],0,output_choice)
#    _, _, poles = ctrl.damp(wing.SS, doprint=False)
#    poles1 = [poles[0], poles[1]]
#    poles2 = [poles[2], poles[3]]
#    plt.plot(np.real(poles1),np.imag(poles1),  linestyle = 'None', color='#000000', marker = 'o', markersize=5)
#    plt.plot(np.real(poles2),np.imag(poles2),  linestyle = 'None', color='#929591', marker = '^', markersize=5)
#    plt.xlabel('Real', fontsize=16)
#    plt.ylabel('Imag', fontsize=16)
#    plt.xlim([-5, 1])
#    plt.title('Pole location at ' + r'$V='+ "{:.1f}".format(V[i]) + 'm/s$ - ' + r'$V_F ='+ "{:.1f}".format(Vf_OL_init) + 'm/s$', fontsize=18)
#    plt.savefig('Results/Video/'+str(i)+'.png')
#plt.plot(np.real(poles1),np.imag(poles1),  linestyle = 'None', color='#000000', marker = 'o', markersize=5, label='Pitch Mode')
#plt.plot(np.real(poles2),np.imag(poles2),  linestyle = 'None', color='#929591', marker = '^', markersize=5, label='Plunge Mode')
#plt.legend(loc='best', fontsize=16)
#plt.xlim([-5, 1])
#plt.savefig('Results/Video/0.png')

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
K = np.array([-0.01, 0.1, -0.001, 0.003])

## Initialization
DV_init = np.concatenate((th, K)) #Design variables

lb_DV = np.array([geom_prop['c']/20, geom_prop['c']/20, geom_prop['h']/20, geom_prop['h']/20, 0, -Kh, -Ka, -Khd, -Kad])
ub_DV = np.array([geom_prop['c']/10, geom_prop['c']/10, geom_prop['h']/10, geom_prop['h']/10, geom_prop['c']-2*geom_prop['c']/10, Kh, Ka, Khd, Kad])

x_init = value2norm(DV_init, lb_DV, ub_DV)
lb = value2norm(lb_DV, lb_DV, ub_DV)
ub = value2norm(ub_DV, lb_DV, ub_DV)

## Reference values
Ref = np.array([m_init, Vf_OL_init])

# Constraints
myConstr = ({'type': 'ineq', 'fun': constraints, 'args':(rho_air, geom_prop, mat_prop, Ref, lb_DV, ub_DV, V, output_choice, aero_coef, EF)})

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

#%% OPTIMIZATION - MINIMIZE - this one works

time_start = perf_counter()
eps = 5e-2

# Optimization
opt = minimize(objective, 
               x_init, 
               args=(rho_air, geom_prop, mat_prop, Ref, lb_DV, ub_DV, V, output_choice, aero_coef, EF, f,),
               method=Met,
               jac = '2-point',
               bounds=Bnd,
               constraints=myConstr,
               tol = 1e-4,
               callback=callbackFunc,
               options={'disp': True, 'eps':eps})

time_end = perf_counter()

time_elapsed = (time_end - time_start)

print("Elapsed time during the whole program in seconds:", time_elapsed) 


#%% ORGANIZE AND PLOT RESULTS 

k = np.arange(0, len(xk), 1)
    
xk_value = []
    
for it in k:
    x = norm2value(xk[it], lb_DV, ub_DV)
    xk_value.append(x)
 
history = {'th1':[], 'th2':[], 'th3':[], 'th4':[], 'th5':[], 'Ka':[], 'Kad':[], 'Kh':[], 'Khd':[], 'm':[], 'Vf_OL':[], 'Vf_CL':[], 'f_obj':[], 'constraint': [], 'EF':[]}

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
    
        
## Mass and flutter velocities for each xk
    [m, Vf_OL, Vf_CL] = mass_velocities(it, aero_coef, history, rho_air, geom_prop, mat_prop, V, output_choice, EF)
        
    history['m'].append(m)
    history['Vf_OL'].append(Vf_OL)
    history['Vf_CL'].append(Vf_CL)
    
    if f==1:
        history['f_obj'].append(history['m'][it]/Ref[0] - history['Vf_CL'][it]/Ref[1])
    if f==2:
        history['f_obj'].append(history['m'][it]/Ref[0])
        
    history['constraint'].append(history['Vf_CL'][it]/Ref[1]-1)

#Plot and save results
case = 'f'+str(f)+'eps'+str(eps)
os.mkdir('Results/'+str(case))
plot_history (range(0,len(xk)), history, case)
save_results(case, history, opt, time_elapsed)
plt.close('all')