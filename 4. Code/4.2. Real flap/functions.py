# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:59:54 2020

@author: Pilar Tagliero
"""

###############################################################################

import numpy as np
from math import pi, sqrt, atan, cos, sin, acos
from AE_model import AE_model_OL, AE_model_CL
import control as ctrl
import matplotlib.pyplot as plt


#def placewingbox(geom_prop):
#    
#    # Unpack data    
#    c = geom_prop['c']
#    EF = geom_prop['EF']
#    t_G = geom_prop['t_G']
#
## Set the flap, then place the wingbox in the remaining space
#
#    t_af = t_G/c # Height of the airfoil divided the chord - fixed - Goland wing
#    
#    x1_c = 1 - EF  # Left wall of the cross section coord. divided the chord
#    
#    x1 = (x1_c)*c # Left wall of the cross section coord.
#    
#    y1_c = t_af*5*(0.2969*sqrt(x1_c)-0.1260*(x1_c)-0.3516*(x1_c)**2+0.2843*(x1_c)**3-0.1036*(x1_c)**4)
#    # Semi height of the left wall of the cross section divided the chord
#    # It is determined by the intersection of the wall with the top of the airfoil
#    
#    y1 = y1_c*c
#    p = [-0.1036, 0, 0.2843, 0, -0.3516, 0, -0.1260, 0.2969, -y1_c/(5*t_af)]
#    p_roots = np.roots(p)
#    
#    p_roots_r = []
#    
#    for root in p_roots:
#        if root.imag==0:
#            p_roots_r.append(root.real**2)
#    
#    x2 = min(p_roots_r)*c
#    
#    cs = (x1-x2) #Structural chord, lenth of the wingbox 
#    
#    h = 2*y1 #[m] height of the section 
#    
#    return [cs, h]


def airfoil(geom_prop, mat_prop):

### OBJECTIVE: Obtain airfoil parameters from beam cross-section gemetrical
# parameters 

### INPUTS: 
# s = semispan
# c = width of wingbox / chord of the airfoil
# h = hight of wingbox
# th = [t1, t2, t3, t4, t5] cross-section shell thickness
# lc = position of the center of the central wall normalized with c
# mat_prop = material properties: rho or mu, E or EI, G or GJ

### Run the code below for input reference:
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#img = mpimg.imread('Beam cross-section.png')
#imgplot = plt.imshow(img)
#plt.show()

### OUTPUTS:
# m = total mass: includes beam (m_b) + flap (m_f) + actuator (m_a)
# I_alpha = moment of inertia relative to z
# w_h = pitch frequency
# w_alpha = plunge frequency
# xg = c.g. x position (distance from the cross-section left wall)        

# Unpack data    
    s = geom_prop['s']
    c = geom_prop['c']
    h = geom_prop['h']
    th = geom_prop['th']
    thf = geom_prop['thf']
    th = np.concatenate(([thf], th))
    ea_b = geom_prop['ea_b']
    EF = geom_prop['EF']
    
    # Chords
    cf = EF*c
    cs = c - cf
 
   
    # Cross-section area 
    hr = h-th[3]-th[4]
    a = np.array([cf*th[0], hr*th[1], hr*th[2], cs*th[3], cs*th[4], hr*th[5]])
    A = sum(a) 
    
#    L = np.array([h, h, cs, cs, h])
#    a = np.multiply(th,L)-np.array([th[1]*(th[3]+th[4]), th[2]*(th[3]+th[4]), 0, 0, th[5]*(th[3]+th[4])])
#    A = sum(a)   
        
# Mass beam and flap

    if 'rho' in mat_prop:
        m = mat_prop['rho']*A*s
    elif 'mu' in mat_prop:
        m = mat_prop['mu']*s
        mat_prop['rho'] = mat_prop['mu']/A
    else:
        print ('Data is missing: Load density (rho) or Linear density (mu)')


# Position of the center of gravity 

    xg = (cs*a[2]+0.5*cs*a[5]+0.5*cs*a[3]+0.5*cs*a[4]+(cs+cf/2)*a[0])/A
    yg = (0.5*h*(a[1]+a[2]+a[5]+a[0])+h*a[4])/A    

# cg = distance of the cg from the midchord
# cg_b = cg normalized with the semichord (b)
# cg = xg - b
# cg/b = xcg/b - 1 
# cg/b = 2*xcg/c - 1 
    cg_b = 2/c*xg-1 
    
#    print(cg_b)
    
# Distance between the aeroelastic axis (ea) and the center of gravity (cg) normalized with the semichord (b)
#    ea_b = (1-EF)*sqrt(EF*(1-EF))/(cs/2*acos(1-2*EF)+2*sqrt(EF*(1-EF)))
    x_alpha = ea_b - cg_b #
    
#    print(x_alpha)
#    print('xg', xg)
#    print('yg', yg)
#    print('A', A)
    
# Second moments of area 

# Iy 

    Iy1 = th[1]**3*hr/12 + a[1]*(xg-th[1]/2)**2
    Iy2 = th[2]**3*hr/12 + a[2]*(cs-th[2]/2-xg)**2
    Iy3 = cs**3*th[3]/12 + a[3]*(0.5*cs-xg)**2
    Iy4 = cs**3*th[4]/12 + a[4]*(0.5*cs-xg)**2
    Iy5 = th[5]**3*hr/12 + a[5]*(0.5*cs-xg)**2
    Iyf = cf**3*th[0]/12 + a[0]*(cs+0.5*cf-xg)**2
    Iy = Iy1+Iy2+Iy3+Iy4+Iy5+Iyf
       
# Ix 
    Ix1 = th[1]*hr**3/12 + a[1]*(0.5*h-yg)**2
    Ix2 = th[2]*hr**3/12 + a[2]*(0.5*h-yg)**2
    Ix3 = cs*th[3]**3/12 + a[3]*(yg-th[3]/2)**2
    Ix4 = cs*th[4]**3/12 + a[4]*(h-th[4]/2-yg)**2
    Ix5 = th[5]*hr**3/12 + a[5]*(0.5*h-yg)**2   
    Ixf = cf*th[0]**3/12 + a[0]*(0.5*h-yg)**2
    Ix = Ix1+Ix2+Ix3+Ix4+Ix5+Ixf
    
# Iz
    Iz = Ix+Iy
    
#    print('Iy', Iy)
#    print('Ix', Ix) 

# Torsional constant (J)
   
#    # OPTION 1: Thin-walled + rectangular in the middle + flap
#    
#    if th[5] < hr:
#        J5 =  hr*th[5]**3*(1/3 - 0.21*(th[5]/hr)*(1-(th[5]**4/(12*hr**4))))     
#    else: 
#        J5 = th[5]*hr**3*(1/3 - 0.21*(hr/th[5])*(1-(hr**4/(12*th[5]**4)))) 
#
#    Jf = cf*th[0]**3*1/3
#    
##    Jf = cf*th[0]**3*(1/3 - 0.21*(th[0]/cf)*(1-(th[0]**4/(12*cf**4))))     
#
#    I = hr/th[1]+hr/th[2]+cs/th[3]+cs/th[4]
#
#    J = 4*(cs*h)**2/I + J5 + Jf
#    
##    print('J',J)
    
    # OPTION 2: All thin-walled
    
    I11 = hr/th[1] + hr/th[5] + cs/(2*th[3]) + cs/(2*th[4]) 
    I22 = hr/th[2] + hr/th[5] + cs/(2*th[3]) + cs/(2*th[4]) 
    I12 = -hr/th[5]
    
    J = 4*(cs/2*h)**2 *(I22+I11-I12)/(I11*I22-I12**2) + 1/3*cf*th[0]**3
#    print('Jsection',J)
    
#     OPTION 3:
    
#    if th[5]/cs < 0.1:
#        I11 = hr/th[1] + hr/th[5] + cs/(2*th[3]) + cs/(2*th[4]) 
#        I22 = hr/th[2] + hr/th[5] + cs/(2*th[3]) + cs/(2*th[4]) 
#        I12 = -hr/th[5]
#        
#        J =  4*(cs/2*h)**2 *(I22+I11-I12)/(I11*I22-I12**2)+ 1/3*cf*th[0]**3   

#    
#    else:
#        if th[5] < hr:
#            J5 =  hr*th[5]**3*(1/3 - 0.21*(th[5]/hr)*(1-(th[5]**4/(12*hr**4))))     
#        else: 
#            J5 = th[5]*hr**3*(1/3 - 0.21*(hr/th[5])*(1-(hr**4/(12*th[5]**4)))) 
#      
#    
#        I = hr/th[1]+hr/th[2]+cs/th[3]+cs/th[4] 
#        
#        J = 4*(cs*h)**2/I  + J5 + 1/3*cf*th[0]**3
    
#    print('J',J)
    
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
# damping of the system assuming quasi steady aerodynamic

### INPUTS: 
# AE_model = State space of the aeroelastic model (can be open or closed-loop)
# V = Range of speed speeds where the frequency and damping are computed 

### OUTPUTS:
# omega = Pulsation for each velocity (rad/s)
# xsi = Damping for each velocity    
# Vf = Flutter velocity: first velocity where damping is (very close to) 
# negative  
        
    for i in range(len(V)):
        model.gen_SS(V[i],0,output_choice)
        wn, xsi, poles = ctrl.damp(model.SS, doprint=False)
        wn_f = []
        if any(x<0 for x in xsi)==True:
            V_f = V[i] # Open-loop flutter speed
            wn_f = [min(wn), max(wn)]
            break

    return [V_f, wn_f]          

    

###############################################################################

def value2norm(V,LL,UL):

    a = np.divide(2,(UL-LL))
    b = np.divide((UL+LL),(UL-LL))
    x = np.multiply(a,V)-b
    
    return x


def norm2value(x,LL,UL):
    
    a = np.divide(2,(UL-LL))
    b = -np.divide((UL+LL),(UL-LL))
    DV = np.divide((x-b),a)
    
    return DV


###############################################################################


def objective (x, *args):
    rho_air, geom_prop, mat_prop, Ref, lb_DV, ub_DV, V, output_choice, aero_coef, f = args
    
    # Recover the design variables
    DV = norm2value(x, lb_DV, ub_DV)
    geom_prop['th'] = DV[0:5]
    K = DV[5:9]
    EF = DV[9]
    
    # Build airfoil with the design variables
    [m, I_alpha, w_h, w_alpha, xg, J, x_alpha] = airfoil (geom_prop, mat_prop)
    
    # Build closed loop
    wing_CL = AE_model_CL(aero_coef, rho_air, geom_prop['ea_b'], geom_prop['c'], geom_prop['s'], m, I_alpha, x_alpha, w_alpha, w_h, output_choice, EF, K)
    
    # Get flutter velocity in closed-loop
    [Vf_CL, wn_f_CL] = V_flutter(wing_CL, V, output_choice)
    
    if f==1:
        f_obj = m/Ref[0] - Vf_CL/Ref[1]
    else:
        f_obj = m/Ref[0] 

    print(f_obj)
    
    return f_obj



def constraints (x, rho_air, geom_prop, mat_prop, Ref, lb_DV, ub_DV, V, output_choice, aero_coef):
    DV = norm2value(x,lb_DV, ub_DV)
    geom_prop['th'] = DV[0:5]
    K = DV[5:9]
    EF = DV[9]
    
    [m, I_alpha, w_h, w_alpha, xg, J, x_alpha] = airfoil (geom_prop, mat_prop)

    wing_CL = AE_model_CL(aero_coef, rho_air, geom_prop['ea_b'], geom_prop['c'], geom_prop['s'], m, I_alpha, x_alpha, w_alpha, w_h, output_choice, EF, K)
    [Vf_CL, wn_f_cl] = V_flutter(wing_CL, V, output_choice)
    
    return Vf_CL/Ref[1]-1


def mass_velocities (j, aero_coef, history, rho_air, geom_prop, mat_prop, V, output_choice):
    th = np.array([history['th1'][j], history['th2'][j], history['th3'][j], history['th4'][j], history['th5'][j]])
    geom_prop['th'] = th
    K = np.array([history['Kh'][j], history['Ka'][j], history['Khd'][j], history['Kad'][j]])
    EF = history['EF'][j]
    
    [m, I_alpha, w_h, w_alpha, xg, J, x_alpha] = airfoil (geom_prop, mat_prop)

    wing_OL = AE_model_OL(aero_coef, rho_air, geom_prop['ea_b'], geom_prop['c'], geom_prop['s'], m, I_alpha, x_alpha, w_alpha, w_h, output_choice, EF)    
    [Vf_OL, wn_f_OL] = V_flutter(wing_OL, V, output_choice)

    wing_CL = AE_model_CL(aero_coef, rho_air, geom_prop['ea_b'], geom_prop['c'], geom_prop['s'], m, I_alpha, x_alpha, w_alpha, w_h, output_choice, EF, K)
    [Vf_CL, wn_f_CL] = V_flutter(wing_CL, V, output_choice)
       
    return m, Vf_OL, Vf_CL


def plot_history (k, history, case): 
    
    plt.figure(1)
    plt.plot(k, history['m'], linestyle='-', marker='p', color='#000000', linewidth=1)
    plt.xlabel('Iterations', fontsize=16)
    plt.title('Mass', fontsize=16)
    plt.grid(True)
    plt.savefig('Results/'+str(case)+'/mass.png')
    plt.show()
    
    plt.figure(2)
    plt.plot(k, history['Vf_OL'], linestyle='-', marker='p', color='#000000', linewidth=1, label='Open Loop (OL)')
    plt.plot(k, history['Vf_CL'], linestyle='-', marker='p', color='#929591', linewidth=1, label='Closed-Loop (CL)')
    plt.plot(k, history['Vf_OL'][0]*np.ones(len(k)), 'k:', linewidth=1, label='Minimum CL velocity allowed')
    plt.xlabel('Iterations', fontsize=16)
    plt.title('Flutter Velocities', fontsize=16)
    plt.legend(loc='best', fontsize=16)
    plt.grid(True)
    plt.savefig('Results/'+str(case)+'/flutter_velocities.png')
    plt.show()
    
    plt.figure(3)
    plt.plot(k, history['f_obj'], linestyle='--', marker='p', color='#000000', linewidth=1)
    plt.xlabel('Iterations', fontsize=16)
    plt.title('Objective Function', fontsize=16)
    plt.grid(True)
    plt.savefig('Results/'+str(case)+'/f_obj.png')
    plt.show()
    
    plt.figure(4)
    plt.plot(k, history['th1'], linestyle='--', marker='p', color='#929591', linewidth=1, label='$t_1$')
    plt.plot(k, history['th2'], linestyle='-', marker='p', color='#000000', linewidth=1, label='$t_2$')
    plt.plot(k, history['th5'], linestyle=':', marker='^', color='#000000', linewidth=1, label='$t_5$') 
    plt.xlabel('Iterations', fontsize=16)
    plt.title('Cross-section Thickness', fontsize=16)
    plt.legend(loc='best', fontsize=16)
    plt.grid(True)
    plt.savefig('Results/'+str(case)+'/th1th2.png')
    plt.show()
    
    plt.figure(5)
    plt.plot(k, history['th4'], linestyle='-', marker='^', color='#000000', linewidth=1, label='$t_4$') 
    plt.plot(k, history['th3'], linestyle='--', marker='p', color='#929591', linewidth=1, label='$t_3$')
    plt.xlabel('Iterations', fontsize=16)
    plt.title('Cross-section Thickness', fontsize=16)
    plt.legend(loc='best', fontsize=16)
    plt.grid(True)
    plt.savefig('Results/'+str(case)+'/th3th4.png')
    plt.show()
        
    
    
#    plt.figure(5)
##    plt.plot(k, history['th1'], linestyle='--', marker='p', color='g', linewidth=1, label='Thickness 1')
#    plt.plot(k, history['th2'], linestyle='-', marker='p', color='b', linewidth=1, label='Thickness 2')
##    plt.plot(k, history['th3'], linestyle='-', marker='p', color='r', linewidth=1, label='Thickness 3')
##    plt.plot(k, history['th4'], linestyle='--', marker='p', color='y', linewidth=1, label='Thickness 4')
##    plt.plot(k, history['th5'], linestyle='--', marker='p', color='m', linewidth=1, label='Thickness 5')
#    plt.xlabel('Iterations', fontsize=16)
#    plt.title('Cross-section Shell Thickness', fontsize=18)
#    plt.legend(loc='best', fontsize=16)
#    plt.grid(True)
#    plt.savefig('Results/'+str(case)+'/th2.png')
#    plt.show()
#    
#    plt.figure(6)
##    plt.plot(k, history['th1'], linestyle='--', marker='p', color='g', linewidth=1, label='Thickness 1')
##    plt.plot(k, history['th2'], linestyle='-', marker='p', color='b', linewidth=1, label='Thickness 2')
#    plt.plot(k, history['th3'], linestyle='-', marker='p', color='r', linewidth=1, label='Thickness 3')
##    plt.plot(k, history['th4'], linestyle='--', marker='p', color='y', linewidth=1, label='Thickness 4')
##    plt.plot(k, history['th5'], linestyle='--', marker='p', color='m', linewidth=1, label='Thickness 5')
#    plt.xlabel('Iterations', fontsize=16)
#    plt.title('Cross-section Shell Thickness', fontsize=18)
#    plt.legend(loc='best', fontsize=16)
#    plt.grid(True)
#    plt.savefig('Results/'+str(case)+'/th3.png')
#    plt.show()
    
#    plt.figure(7)
##    plt.plot(k, history['th1'], linestyle='--', marker='p', color='g', linewidth=1, label='Thickness 1')
##    plt.plot(k, history['th2'], linestyle='-', marker='p', color='b', linewidth=1, label='Thickness 2')
##    plt.plot(k, history['th3'], linestyle='-', marker='p', color='r', linewidth=1, label='Thickness 3')
#    plt.plot(k, history['th4'], linestyle='--', marker='p', color='y', linewidth=1, label='Thickness 4')
##    plt.plot(k, history['th5'], linestyle='--', marker='p', color='m', linewidth=1, label='Thickness 5')
#    plt.xlabel('Iterations', fontsize=16)
#    plt.title('Cross-section Shell Thickness', fontsize=18)
#    plt.legend(loc='best', fontsize=16)
#    plt.grid(True)
#    plt.savefig('Results/'+str(case)+'/th4.png')
#    plt.show()    
    
#    plt.figure(8)
##    plt.plot(k, history['th1'], linestyle='--', marker='p', color='g', linewidth=1, label='Thickness 1')
##    plt.plot(k, history['th2'], linestyle='-', marker='p', color='b', linewidth=1, label='Thickness 2')
##    plt.plot(k, history['th3'], linestyle='-', marker='p', color='r', linewidth=1, label='Thickness 3')
##    plt.plot(k, history['th4'], linestyle='--', marker='p', color='y', linewidth=1, label='Thickness 4')
#    plt.plot(k, history['th5'], linestyle='--', marker='p', color='m', linewidth=1, label='Thickness 5')
#    plt.xlabel('Iterations', fontsize=16)
#    plt.title('Cross-section Shell Thickness', fontsize=18)
#    plt.legend(loc='best', fontsize=16)
#    plt.grid(True)
#    plt.savefig('Results/'+str(case)+'/th5.png')
#    plt.show()        
    
    plt.figure(6)
    plt.plot(k, history['Kh'], linestyle='-', marker='p', color='#929591', linewidth=1, label=r'$K_{h}$')
    plt.plot(k, history['Ka'], linestyle='--', marker='p', color='#000000', linewidth=1, label=r'$K_{\alpha}$')
    plt.xlabel('Iterations', fontsize=16)
    plt.title('Proportional Control Gains', fontsize=16)
    plt.legend(loc='best', fontsize=16)
    plt.grid(True)
    plt.savefig('Results/'+str(case)+'/Kp.png')
    plt.show()

#    plt.figure(10)
##    plt.plot(k, history['Kh'], linestyle='-', marker='p', color='b', linewidth=1, label='Gain alpha dot')
#    plt.plot(k, history['Ka'], linestyle='--', marker='p', color='g', linewidth=1, label='Ka')
##    plt.plot(k, history['Khd'], linestyle='--', marker='p', color='y', linewidth=1, label='Gain h dot')
##    plt.plot(k, history['Kad'], linestyle='-', marker='p', color='r', linewidth=1, label='Gain alpha')
#    plt.xlabel('Iterations', fontsize=16)
#    plt.title('Derivative Control Gains', fontsize=18)
#    plt.legend(loc='best', fontsize=16)
#    plt.grid(True)
#    plt.savefig('Results/'+str(case)+'/Ka.png')
#    plt.show()
    
    plt.figure(7)
    plt.plot(k, history['Khd'], linestyle='-', marker='p', color='#929591', linewidth=1, label=r'$K_{\dot{h}}$')
    plt.plot(k, history['Kad'], linestyle='--', marker='p', color='#000000', linewidth=1, label=r'$K_{\dot{\alpha}}$')
    plt.xlabel('Iterations', fontsize=16)
    plt.title('Derivative Control Gains', fontsize=16)
    plt.legend(loc='best', fontsize=16)
    plt.grid(True)
    plt.savefig('Results/'+str(case)+'/Kd.png')
    plt.show()

#    plt.figure(12)
##    plt.plot(k, history['Kh'], linestyle='-', marker='p', color='b', linewidth=1, label='Gain alpha dot')
##    plt.plot(k, history['Ka'], linestyle='--', marker='p', color='g', linewidth=1, label='Gain alpha')
##    plt.plot(k, history['Khd'], linestyle='--', marker='p', color='y', linewidth=1, label='Gain h dot')
#    plt.plot(k, history['Kad'], linestyle='-', marker='p', color='r', linewidth=1, label='Kad')
#    plt.xlabel('Iterations', fontsize=16)
#    plt.title('Control Gain', fontsize=18)
#    plt.legend(loc='best', fontsize=16)
#    plt.grid(True)
#    plt.savefig('Results/'+str(case)+'/Kad.png')
#    plt.show()
    
 
    plt.figure(8)
    plt.plot(k, history['EF'], linestyle='-', marker='p', color='#929591', linewidth=1)
    plt.xlabel('Iterations', fontsize=16)
    plt.title('Ratio of flap chord to total chord', fontsize=16)
    plt.grid(True)
    plt.savefig('Results/'+str(case)+'/EF.png')  
    plt.show() 
    
    
###############################################################################    
    
def save_results(case, history, opt, time_elapsed):
    with open('Results/'+str(case)+'/Variables.txt', 'w') as file:
        file.write('Case:'+str(case)+'   '+'Time:'+str(time_elapsed))
        file.write('\n')
        file.write(str(opt))
        file.write('\n')
        for var in history:
            file.write(str(var)+': '+str(history[var]))
            file.write('\n')

###############################################################################
#    def controller (type):
#        
#        if type == 'feedback_gain':
#            # d_max = RL/4f, f:max frequency at flutter velocity
#            # d_max = 0.5*Kh*d_max 
#            # d_max = 0.5*Ka*alpha_max
#            # d_max/(f/2*pi) = 0.5*Khd*d_max 
#            # d_max/(f/2*pi) = 0.5*Kad*alpha_max
#                       
#    elif type =
        