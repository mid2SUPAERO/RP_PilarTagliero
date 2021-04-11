# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 19:43:58 2021

@author: p.tagliero
"""

import numpy as np
import matplotlib.pyplot as plt

c = 1.829 
EF = 0.2
h = 0.5

th1 = c*(1-EF)/10
th2 = c*(1-EF)/10
th3 = h/10
th4 = h/10
thf = (EF*c)/10

Jtw = [] #thin-walled

Je = [] #Emmeline

th5 = np.arange(1e-3, 1.17056, 0.001)

for th5_i in th5:
    th = np.array([thf, th1, th2, th3, th4, th5_i])
    cf = EF*c
    cs = c - cf
    hr = h-th[3]-th[4]

    I11 = hr/th[1] + hr/th[5] + cs/(2*th[3]) + cs/(2*th[4]) 
    I22 = hr/th[2] + hr/th[5] + cs/(2*th[3]) + cs/(2*th[4]) 
    I12 = -hr/th[5]
    
    Jtw_i =  4*(cs/2*h)**2 *(I22+I11-I12)/(I11*I22-I12**2) + 1/3*cf*th[0]**3
    Jtw.append(Jtw_i)
    
    ####
    
    if th[5] < hr:
        J5 =  hr*th[5]**3*(1/3 - 0.21*(th[5]/hr)*(1-(th[5]**4/(12*hr**4))))     
    else: 
        J5 = th[5]*hr**3*(1/3 - 0.21*(hr/th[5])*(1-(hr**4/(12*th[5]**4)))) 
     
    I = hr/th[1]+hr/th[2]+cs/th[3]+cs/th[4] 
        
    Je_i = 4*(cs*h)**2/I  + J5 
    
    Je.append(Je_i)
    
Jrect = 0.263*cs*h**3 
    
plt.figure(1)
plt.plot(th5, Jtw, linestyle='-', color='#929591', linewidth=2, label='$J = J_{tw2} + J_{f}$')
plt.plot(th5, Je, linestyle='-',  color='#000000', linewidth=2, label='$J = J_{tw1} +  J_{5}$')
plt.plot(th5[-1], Jrect, linestyle='', color='#000000',  marker='*', label='$J = J_{solid}$')
plt.xlabel('$t_5$', fontsize=16)
plt.title('$J$', fontsize=16)
plt.legend(loc='best', fontsize=16)
plt.grid(True)
plt.show()



#############################################################################

l_s = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2, 2.5, 3, 4]
Data = [0.1928, 0.1973, 0.2006, 0.2031, 0.205, 0.2064, 0.2074, 0.2086, 0.2093, 0.2099, 0.2101, 0.2101]
Fit = []

for i in l_s:
    Fit_i = 0.21*(1-1/12*1/i**4)
    Fit.append(Fit_i)

plt.figure(2)
plt.plot(l_s, Fit, linestyle='-', color='#929591', linewidth=2, label=r'$0.21(1-1/12(s/l)^4)$')
plt.plot(l_s, Data, linestyle='', color='#000000',  marker='*')
plt.xlabel(r'$l/s$', fontsize=16)
plt.title('$\Phi$ - Coeficient of edge loss', fontsize=16)
plt.legend(loc='best', fontsize=16)
plt.grid(True)
plt.show()

