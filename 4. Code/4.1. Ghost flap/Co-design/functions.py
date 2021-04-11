# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:59:54 2020

@author: Pilar Tagliero
"""

###############################################################################

import numpy as np
from math import pi, sqrt
from AE_model import AE_model_OL, AE_model_CL
import control as ctrl
import matplotlib.pyplot as plt



def airfoil(geom_prop, mat_prop):

### OBJECTIVE: Obtain airfoil parameters from beam cross-section gemetrical
# parameters 

### INPUTS: 
# s = semispan
# c = width of wingbox / chord of the airfoil
# h = hight of wingbox
# th = [t1, t2, t3, t4, t5] cross-section shell thickness
# mat_prop = material properties: rho or mu, E or EI, G or GJ

### OUTPUTS:
# m = total mass
# I_alpha = mass moment of inertia relative to z
# w_h = plunge frequency
# w_alpha = pitch frequency
# xg = c.g. x position (distance from the cross-section left wall)        
    
    
# Unpack data    
    s = geom_prop['s']
    c = geom_prop['c']
    h = geom_prop['h']
    th = geom_prop['th']
    ea_b = geom_prop['ea_b']

# Cross-section area 
    l = c
    L = np.array([h, h, l, l, h])
    a = np.multiply(th,L)-np.array([th[0]*(th[2]+th[3]), th[1]*(th[2]+th[3]), 0, 0, th[4]*(th[2]+th[3])])
    A = sum(a)   
        
# Mass
    if 'rho' in mat_prop:
        m = mat_prop['rho']*A*s
    elif 'mu' in mat_prop:
        m = mat_prop['mu']*s
        mat_prop['rho'] = mat_prop['mu']/A
    else:
        print ('Data is missing: Load density (rho) or Linear density (mu)')

# Position of the center of gravity 
    xg = (l*a[1]+0.5*l*a[4]+0.5*l*a[2]+0.5*l*a[3])/A
    yg = (0.5*h*(a[0]+a[1]+a[4])+h*a[2])/A

# cg = distance of the cg from the midchord
# cg_b = cg normalized with the semichord (b)
# cg = xg - b
# cg/b = xcg/b - 1 
# cg/b = 2*xcg/c - 1 
    cg_b = 2/c*xg-1

# Distance between the aeroelastic axis (ea) and the center of gravity (cg) normalized with the semichord (b)
    x_alpha = ea_b - cg_b


# Iy 
    hr = h-th[3]-th[2]
    Iy1 = th[0]**3*hr/12 + a[0]*(xg-th[0]/2)**2
    Iy2 = th[1]**3*hr/12 + a[1]*(c-th[1]/2-xg)**2
    Iy3 = c**3*th[2]/12 + a[2]*(0.5*c-xg)**2
    Iy4 = c**3*th[3]/12 + a[3]*(0.5*c-xg)**2
    Iy5 = th[4]**3*h/12 + a[4]*(0.5*c-xg)**2
    Iy = Iy1+Iy2+Iy3+Iy4+Iy5

# Ix 
    Ix1 = th[0]*hr**3/12 + a[0]*(0.5*h-yg)**2
    Ix2 = th[1]*hr**3/12 + a[1]*(0.5*h-yg)**2
    Ix3 = c*th[2]**3/12 + a[2]*(yg-th[2]/2)**2
    Ix4 = c*th[3]**3/12 + a[3]*(h-th[3]/2-yg)**2
    Ix5 = th[4]*h**3/12 + a[4]*(0.5*h-yg)**2   
    Ix = Ix1+Ix2+Ix3+Ix4+Ix5

# Iz: relative to an axis perpendicular to the wingbox cross-section
    Iz = Ix+Iy

# Torsional constant (J) 
    if th[4] < hr:
        I = hr/th[0]+hr/th[1]+l/th[2]+l/th[3]
        J = 4*(l*h)**2/I+hr*th[4]**3*((1/3)-0.21*(th[4]/hr)*(1-(th[4]**4/(12*hr**4))))
    else: 
        I = hr/th[0]+hr/th[1]+l/th[2]+l/th[3] 
        J = 4*(l*h)**2/I+th[4]*hr**3*((1/3)-0.21*(hr/th[4])*(1-(hr**4/(12*th[4]**4))))

# Moment of inertia 
    I_alpha = (m/A)*Iy
    
# Frequency
    X1 = 1.875 # first non zero sol of cos(x)*cosh(x)+1 = 0 
        
    # Flexural mode
    if 'EI' in mat_prop:
        w_h = sqrt((X1/s)**4*(mat_prop['EI'])/(A*mat_prop['rho']))
    else:
        w_h = sqrt((X1/s)**4*(mat_prop['E']*Ix)/(A*mat_prop['rho']))    
    
    if 'GJ' in mat_prop:
        w_alpha = (pi/(2*s))*sqrt(mat_prop['GJ']/(mat_prop['rho']*Iz))
    else:
        w_alpha = (pi/(2*s))*sqrt(mat_prop['G']*J/(mat_prop['rho']*Iz))


    return [m, I_alpha, w_h, w_alpha, xg, J, x_alpha]


###############################################################################
    
def V_flutter(model,V,output_choice): 
    
# OBJECTIVE: obtain flutter velocity by means of computing the frequency and 
# damping of the system assuming quasisteady aerodynamics

### INPUTS: 
# AE_model = State space of the aeroelastic model (can be open or closed-loop)
# V = Range of speed speeds where the frequency and damping are computed 

### OUTPUTS:
# omega = Pulsation for each velocity (rad/s)
# xsi = Damping for each velocity    
# Vf = Flutter velocity: first velocity where damping is (very close to) 0  
        
    for i in range(len(V)):
        model.gen_SS(V[i],0,output_choice)
        wn, xsi, poles = ctrl.damp(model.SS, doprint=False)
        wn_f = []
        if any(x<0 for x in xsi)==True:
            V_f = V[i] # Flutter speed
            wn_f = [min(wn), max(wn)]
            break
        SS_f = model.SS
        _,_, poles_f = ctrl.damp(SS_f, doprint=False) 
        
    return [V_f, wn_f, SS_f, poles_f]    
   

###############################################################################

def value2norm(V,LL,UL):
    
# OBJECTIVE: Normalize design variables so as upper_boundaries are all 1 
# and lower boundaries are all -1. 

### INPUTS: 
# V = value to normalize
# UL = upper limit
# LL = lower limit 

### OUTPUTS:
# x = normalized variables

    a = np.divide(2,(UL-LL))
    b = np.divide((UL+LL),(UL-LL))
    x = np.multiply(a,V)-b
    
    return x

###############################################################################

def norm2value(x,LL,UL):

# OBJECTIVE: Recover real values of the design variables from normalized ones    
    
    a = np.divide(2,(UL-LL))
    b = -np.divide((UL+LL),(UL-LL))
    DV = np.divide((x-b),a)
    
    return DV

###############################################################################
    
def objective (x, *args):
    rho_air, geom_prop, mat_prop, Ref, lb_DV, ub_DV, V, output_choice, aero_coef, EF, f = args
    
    # Recover the design variables
    DV = norm2value(x, lb_DV, ub_DV)
    geom_prop['th'] = DV[0:5]
    K = DV[5:9]
    
    # Build airfoil with the design variables
    [m, I_alpha, w_h, w_alpha, xg, J, x_alpha] = airfoil (geom_prop, mat_prop)

    # Build closed loop
    wing_CL = AE_model_CL(aero_coef, rho_air, geom_prop['ea_b'], geom_prop['c'], geom_prop['s'], m, I_alpha, x_alpha, w_alpha, w_h, output_choice, EF, K)

    # Get flutter velocity in closed-loop
    [Vf_CL, _,_,_] = V_flutter(wing_CL, V, output_choice)
   
    if f==1:
        f_obj = m/Ref[0] - Vf_CL/Ref[1]
    if f==2:
        f_obj = m/Ref[0] 

    print('F:', f_obj)

#    f_obj = m/Ref[0] - Vf_CL/Ref[1]
    
#    f_obj = 0.8*model.m/Ref[0] - 0.2*Vf_CL/Ref[1]
    
    return f_obj

def constraints (x, rho_air, geom_prop, mat_prop, Ref, lb_DV, ub_DV, V, output_choice, aero_coef, EF):
    DV = norm2value(x, lb_DV, ub_DV)
    geom_prop['th'] = DV[0:5]
    K = DV[5:9]
    
    [m, I_alpha, w_h, w_alpha, xg, J, x_alpha] = airfoil (geom_prop, mat_prop)
    
    wing_CL = AE_model_CL(aero_coef, rho_air, geom_prop['ea_b'], geom_prop['c'], geom_prop['s'], m, I_alpha, x_alpha, w_alpha, w_h, output_choice, EF, K)
    [Vf_CL, _, _, _] = V_flutter(wing_CL, V, output_choice)
    
    print('C:', Vf_CL/Ref[1]-1)
    
    return Vf_CL/Ref[1]-1


def mass_velocities (j, aero_coef, history, rho_air, geom_prop, mat_prop, V, output_choice, EF):
    th = np.array([history['th1'][j], history['th2'][j], history['th3'][j], history['th4'][j], history['th5'][j]])
    geom_prop['th'] = th
    K = np.array([history['Kh'][j], history['Ka'][j], history['Khd'][j], history['Kad'][j]])
    
    [m, I_alpha, w_h, w_alpha, xg, J, x_alpha] = airfoil (geom_prop, mat_prop)

    wing_OL = AE_model_OL(aero_coef, rho_air, geom_prop['ea_b'], geom_prop['c'], geom_prop['s'], m, I_alpha, x_alpha, w_alpha, w_h, output_choice, EF)    
    [Vf_OL, _, _, _]= V_flutter(wing_OL, V, output_choice)

    wing_CL = AE_model_CL(aero_coef, rho_air, geom_prop['ea_b'], geom_prop['c'], geom_prop['s'], m, I_alpha, x_alpha, w_alpha, w_h, output_choice, EF, K)
    [Vf_CL, _, _, _] = V_flutter(wing_CL, V, output_choice)
       
    return m, Vf_OL, Vf_CL


def plot_history (k, history, case): 
    
    plt.figure(1)
    plt.plot(k, history['m'], linestyle='-', marker='p', color='#000000', linewidth=1)
    plt.xlabel('Iterations')
    plt.title('Mass')
    plt.grid(True)
    plt.savefig('Results/'+str(case)+'/mass.png')
    plt.show()
    
    plt.figure(2)
    plt.plot(k, history['Vf_OL'], linestyle='-', marker='p', color='#000000', linewidth=1, label='Open Loop (OL)')
    plt.plot(k, history['Vf_CL'], linestyle='-', marker='p', color='#929591', linewidth=1, label='Closed-Loop (CL)')
    plt.plot(k, history['Vf_OL'][0]*np.ones(len(k)), 'k:', linewidth=1, label='Minimum CL velocity allowed')
    plt.xlabel('Iterations')
    plt.title('Flutter Velocities')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('Results/'+str(case)+'/flutter_velocities.png')
    plt.show()
    
    plt.figure(3)
    plt.plot(k, history['f_obj'], linestyle='--', marker='p', color='#000000', linewidth=1)
    plt.xlabel('Iterations')
    plt.title('Objective Function')
    plt.grid(True)
    plt.savefig('Results/'+str(case)+'/f_obj.png')
    plt.show()
    
    plt.figure(4)
    plt.plot(k, history['th1'], linestyle='--', marker='p', color='#929591', linewidth=1, label='$t_1$')
    plt.plot(k, history['th2'], linestyle='-', marker='p', color='#000000', linewidth=1, label='$t_2$')
    plt.plot(k, history['th5'], linestyle=':', marker='^', color='#000000', linewidth=1, label='$t_5$') 
    plt.xlabel('Iterations')
    plt.title('Cross-section Thickness')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('Results/'+str(case)+'/th1th2.png')
    plt.show()
    
    plt.figure(5)
    plt.plot(k, history['th4'], linestyle='-', marker='^', color='#000000', linewidth=1, label='$t_4$') 
    plt.plot(k, history['th3'], linestyle='--', marker='p', color='#929591', linewidth=1, label='$t_3$')
    plt.xlabel('Iterations')
    plt.title('Cross-section Thickness')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('Results/'+str(case)+'/th3th4.png')
    plt.show()
    
    plt.figure(6)
    plt.plot(k, history['Kh'], linestyle='-', marker='p', color='#929591', linewidth=1, label=r'$K_{h}$')
    plt.plot(k, history['Ka'], linestyle='--', marker='p', color='#000000', linewidth=1, label=r'$K_{\alpha}$')
    plt.xlabel('Iterations')
    plt.title('Proportional Control Gains')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('Results/'+str(case)+'/Kp.png')
    plt.show()
    
    plt.figure(7)
    plt.plot(k, history['Khd'], linestyle='-', marker='p', color='#929591', linewidth=1, label=r'$K_{\dot{h}}$')
    plt.plot(k, history['Kad'], linestyle='--', marker='p', color='#000000', linewidth=1, label=r'$K_{\dot{\alpha}}$')
    plt.xlabel('Iterations')
    plt.title('Derivative Control Gains')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('Results/'+str(case)+'/Kd.png')
    plt.show()   
    
    
def save_results(case, history, opt, time_elapsed):
    with open('Results/'+str(case)+'/Variables.txt', 'w') as file:
        file.write('Case:'+str(case)+'   '+'Time:'+str(time_elapsed))
        file.write('\n')
        file.write(str(opt))
        file.write('\n')
        for var in history:
            file.write(str(var)+': '+str(history[var]))
            file.write('\n')

        