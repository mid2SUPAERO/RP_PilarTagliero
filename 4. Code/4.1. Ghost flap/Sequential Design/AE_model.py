# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 11:44:02 2020

@author: Pilar Tagliero
"""

### OBJECTIVE: Create the aerodynamic model. Return its space state 

import numpy as np
from math import sqrt, pi, acos
import control as ctrl
from scipy.special import kv

# Objective: Obtain a state space in open loop
class AE_model_OL:
    def __init__(self, aero_coef, rho_air, ea_b, c, s, m, I_alpha, x_alpha, w_alpha, w_h, output_choice, EF):
        
        self.rho = rho_air
        self.s = s
        self.c = c
        self.a = ea_b
        self.b = c/2
        self.m = m
        self.I_alpha = I_alpha
        self.x_alpha = x_alpha

        self.w_alpha = w_alpha
        self.w_h = w_h
        
        self.k_alpha = w_alpha**2*I_alpha
        self.k_h = w_h**2*m
        
        
        self.cl_alpha = aero_coef['cl_alpha']
        self.cm_alpha = aero_coef['cm_alpha']
        
        self.c_h = 2*0.01*sqrt(self.k_h/self.m)*self.m
        self.c_alpha = 2*0.01*sqrt(self.k_alpha/I_alpha)*I_alpha
        self.output_choice = output_choice
        
        # Galuert
        self.EF = EF
        self.cl_beta = aero_coef['cl_alpha']/pi*(acos(1-2*self.EF)+2*sqrt(self.EF*(1-self.EF)))
        self.cm_beta = -aero_coef['cl_alpha']/pi*(1-self.EF)*sqrt(self.EF*(1-self.EF))
        
        # gen_aero_quasisteady
        self.q = 0
        self.Q = 0
        self.Q_dot = 0
        self.U = 0
        
        self.w = 0 
        
        # gen_matrix
        self.M = 0
        self.D_struct = 0
        self.K_struct = 0
        self.K_aero = 0
        self.D_aero = 0
        self.Cc = 0
        self.Cg = 0
        self.SS = 0
        
        # state space
        self.A = 0
        self.B = 0
        self.C = 0
        self.D = 0
        
                
    def gen_aero_quasisteady(self, U):
        self.U = U
        self.q = (1/2)*self.rho*self.U**2
        # Aerodynamic Stiffness
        self.Q = self.q*2*self.b*np.matrix([[0, -self.cl_alpha], [0, self.b*self.cm_alpha]])
        # Aerodynamic Damping
        self.Q_dot = self.q*2*self.b*np.matrix([[-self.cl_alpha/self.U, -(self.cl_alpha/self.U)*((1/2)-self.a)*self.b],\
                [-self.b*self.cm_alpha/self.U, (self.cm_alpha/self.U)*((1/2)-self.a)*self.b**2]])
    
        
    def gen_aero_unsteady(self,U,w):
        self.U = U
        self.w = w
        k = w*self.b/self.U
        # Theodorsen function
        # kv is the modified Bessel function of the second kind Kν(z)
        Theo = kv(1,1j*k)/(kv(0,1j*k)+kv(1,1j*k))
        F = Theo.real
        G = Theo.imag
        
        Lz  = 2*np.pi*(-k**2/2-G*k)
        Lzd = 2*np.pi*F
        Lt  = 2*np.pi*(k**2*self.a/2+F-G*k*(1/2-self.a))
        Ltd = 2*np.pi*(1/2+F*(1/2-self.a)+G/k)
        
        Mz  = 2*np.pi*(-k**2*self.a/2-k*(self.a+1/2)*G)
        Mzd = 2*np.pi*(self.a+1/2)*F
        Mt  = 2*np.pi*(k**2/2*(1/8+self.a**2)+F*(self.a+1/2)-k*G*(self.a+1/2)*(1/2-self.a))
        Mtd = 2*np.pi*(-k/2*(1/2-self.a)+k*F*(self.a+1/2)*(1/2-self.a)+G/k*(self.a+1/2))
        
        self.Q = self.rho*self.U**2*np.matrix([[-Lz, -Lt*self.b],[Mz*self.b, Mt*self.b**2]])
        self.Q_dot = self.rho*self.U*np.matrix([[-Lzd*self.b, -Ltd*self.b**2],[Mzd*self.b**2, Mtd*self.b**3]])
        
        
    def gen_matrix(self,U,w): 
    # Generate structural & aerodynamical model matrix from U (flow velocity) 
    # and w = reduced frequency for unsteady aerodynamics (or 0 if quasisteady)
        self.U = U
        self.w = w
        self.q = (1/2)*self.rho*self.U**2
        # Structural Inertia
        self.M = np.matrix([[self.m, self.m*self.x_alpha*self.b],[self.m*self.x_alpha*self.b, self.I_alpha]])
        # Structural Damping 
        self.D_struct = np.matrix([[self.c_h, 0], [0, self.c_alpha]])       
        # Structural Stiffness
        self.K_struct = np.matrix([[self.k_h, 0], [0, self.k_alpha]])
        # Aerodynamic matrix
        if self.w==0:
            self.gen_aero_quasisteady(self.U)
        else:
            self.gen_aero_unsteady(self.U,self.w)   
        # Control matrix
        self.Cc = self.q*2*self.b*np.array([[-self.cl_beta],[self.b*self.cm_beta]])
        # Gust matrix
        self.Cg = self.q*2*self.b*np.array([[-self.cl_alpha],[self.b*self.cm_alpha]])
        
    
    def gen_SS(self, U, w, output_choice):  
    # Generate the space system formulation of the problem
    
    # Output choice is a row vector that allows choosing the output variables.
    # 1, 2, 3, 4 : h, alpha, h dot, alpha dot
    # 1, 2, 3, 4, 5: h, alpha, h dot, alpha dot, h dot dot
    # 1, 2, 3, 4, 5, 6: h, alpha, h dot, alpha dot, h dot dot, alfa dot dot
        self.U = U
        self.w = w
        self.output_choice = output_choice
        
        self.gen_matrix(self.U, self.w)    
        M_inv = np.linalg.inv(self.M)
       
        # Matrices
        self.A = np.block([[np.zeros((2,2)), np.eye(2)], [np.dot(M_inv,(self.Q-self.K_struct)), np.dot(M_inv,(self.Q_dot-self.D_struct))]])
        self.B = np.block([[0], [0], [np.dot(M_inv,self.Cc)]])
#       M_gust = np.block([[0], [0], [np.dot(M_inv,self.Cg)]])
        
        n_out = sum(self.output_choice)
        self.C = np.zeros((n_out,np.size(self.A,1)))
        self.D = np.zeros((n_out,np.size(self.B,1))) #outputs, columns of B
        
        for i in range(len(self.output_choice)):
            if i < n_out and self.output_choice[i] == 1:
                self.C[i,i] = 1
            if i > n_out and self.output_choice[i] == 1:
                newrowC = np.zeros((1, np.size(self.A,1)))
                newrowC[1,i] = 1
                self.C = np.vstack([self.C, newrowC])
                newrowD = np.zeros((1, np.size(self.B,1)))
                self.D = np.vstack([self.D, newrowD])                     
                n_out = n_out + 1 
        # Space state
        self.SS = ctrl.ss(self.A, self.B, self.C, self.D)

    
class AE_model_CL(AE_model_OL):
# Objective: Close the loop with state feedback controller     
    def __init__(self, aero_coef, rho_air, ea_b, c, s, m, I_alpha, x_alpha, w_alpha, w_h, output_choice, EF, K):
        super().__init__(aero_coef, rho_air, ea_b, c, s, m, I_alpha, x_alpha, w_alpha, w_h, output_choice, EF)
        self.K = K # feedback gain
    
    def gen_SS(self, U, w, output_choice):
        super().gen_SS(U, w, output_choice)
        self.SS = ctrl.feedback(self.SS,self.K) #by default, negative feedback
        [self.A,self.B,self.C,self.D] = ctrl.ssdata(self.SS)
    
        
        

            

        
        
        
        
        