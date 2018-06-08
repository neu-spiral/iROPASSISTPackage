#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 21:04:52 2017

@author: veysiyildiz
"""

import numpy as np

def k_one_dim(x, kernel_type='Gaussian'):
    """zero mean-unit width- one dimentional kernel function
    kernel_type= 'Gaussian' (default)
    """
    k=0.
    if kernel_type=='Gaussian':
        """ gaussian kernel is calculated according to following formula
        
        k(x)= (1/((2pi)^1/2))e^(-x^2 / 2)
        
        missing a calculation in truncted function
        """
        if x<-3. or x>3.:
            k=0.
        else:
            k= (np.exp(-x**2/2)/np.sqrt(2*np.pi))
    else:
        print "I dont know how to calculate this Kernel. Please update k_one_dim(x, kernel_type='Gaussian') and required derivative functions"
    return k
def first_der_k_one_dim(x,kernel_type='Gaussian'):
    """ first derivative of k(x) function
        kernel_type= 'Gaussian' (default)
    """
    if kernel_type=='Gaussian':
        """ first derivative of gaussian kernel is calculated according to following formula
        
        k`(x)= -x* k(x)  where k(x) is one dim gaussian kernel 
        """
        g= -x * k_one_dim(x, kernel_type='Gaussian')
    else:
        print "I dont know how to calculate this Kernel. Please update k_one_dim(x, kernel_type='Gaussian') and required derivative functions"
    return g

def sec_der_k_one_dim(x,kernel_type='Gaussian'):
    """ second derivative of k(x) function
    """
    if kernel_type=='Gaussian':
        """ first derivative of gaussian kernel is calculated according to following formula
    
        k``(x)= -k(x)+ x^2* k(x)  where k(x) is one dim gaussian kernel 
        """
        h= -k_one_dim(x, kernel_type='Gaussian') + (x**2)*k_one_dim(x, kernel_type='Gaussian')
    else:
        print "I dont know how to calculate this Kernel. Please update k_one_dim(x, kernel_type='Gaussian') and required derivative functions"
    return h

def K_mulvar_I_Sca(x, kernel_type='Gaussian'):
    """ Multivariate seperable and isotropic, identity scale kernel( R^d--> R)
    K(x)= Π_(l=1)^(dim) ((k_one_dim(x_l)))
    """
    K=1.
    for a in x:
        K=K*k_one_dim(a, kernel_type)

    return K

def first_der_K_mulvar(x, kernel_type='Gaussian'):
    """ First derivative of Multivariate seperable and isotropic, identity scale kernel( R^d--> R)
    ∇_c(K(x))=first_der_k_one_dim(x_c)*(Π_(l=1&& l!=c)^(dim) ((k_one_dim(x_l))))    
    """
    dim=x.size
    loop_range=range(dim)            #dimension of x 
    grad_K=np.ones(shape=(1,dim))
    for c in loop_range:
        for l in loop_range:
            if c==l:
                grad_K[0,c]= grad_K[0,c]*first_der_k_one_dim(x[l], kernel_type)
            else:

                grad_K[0,c]= grad_K[0,c]*k_one_dim(x[l], kernel_type)                    
    return grad_K

def sec_der_K_mulvar(x, kernel_type='Gaussian'):
    """ Second derivative of Multivariate seperable and isotropic, identity scale kernel( R^d--> R)
    ∇_rc^2(K(x))=dirac(rc)*sec_der_k_one_dim(x_c)*(Π_(l=1&& l!=r,c)^(dim)((k_one_dim(x_l)))) +  \
    (1-dirac(rc))*first_der_k_one_dim(x_c)*first_der_k_one_dim(x_r)*(Π_(l=1&& l!=r,c)^(dim) ((k_one_dim(x_l))))    
    """    
    dim=x.size #dimension of x 
    loop_range=range(dim)            
    hes_K=np.ones(shape=(dim,dim))
    for r in loop_range:
        for c in loop_range:
            if r==c:                
                for l in loop_range:
                    if r==l:
                        hes_K[r,c]= hes_K[r,c]*sec_der_k_one_dim(x[l], kernel_type )
                    else:
                        hes_K[r,c]= hes_K[r,c]*k_one_dim(x[l], kernel_type )     
            else:
                for l in loop_range:
                    if np.logical_or(r==l,c==l):
                        hes_K[r,c]= hes_K[r,c]*first_der_k_one_dim(x[l], kernel_type )
                    else:
                        hes_K[r,c]= hes_K[r,c]*k_one_dim(x[l], kernel_type )                      
    return hes_K
    
def K_S(x,S, kernel_type='Gaussian'):
    """ Multivariate seperable kernel with diagonal scale(inverse-width) matrix ( R^d--> R)
    K_S(x)= beta(S)* K_mulvar_I_Sca(Sx)
        where  beta(S) is a normalization factor to make (integral of K_S(x) over R^d) =1
        so
        beta(S)= |S|
    """       
    beta_S=np.linalg.det(S)
    K_S=beta_S*K_mulvar_I_Sca(S*x, kernel_type)
    return K_S

def first_der_K_S(x,S, kernel_type='Gaussian'):
    """ First derivative of Multivariate seperable kernel with diagonal scale(inverse-width) matrix ( R^d--> R)
    ∇(K_S(x))=|S| * first_der_K_mulvar(S*x) * S
    """   
    return np.linalg.det(S)* np.mat(first_der_K_mulvar(S*x, kernel_type)) * S

def sec_der_K_S(x,S, kernel_type='Gaussian'):
    """ First derivative of Multivariate seperable kernel with diagonal scale(inverse-width) matrix ( R^d--> R)
    ∇(K_S(x))=|S| * S^T * sec_der_K_mulvar(S*x) * S
    """  
    return np.linalg.det(S) * np.transpose(S) * np.mat(sec_der_K_mulvar(S*x, kernel_type)) * S
    


def fx(w,S,X_i,x, kernel_type='Gaussian'):
    """
    Return Kernel Density value of a point accordung to following function
    
    f(x)= Σ_(i=1)^n w_i*K_S_i(x-X_i)
    """
    fx=0.
    for i in range(len(w)):
       fx=fx + w[i]* K_S((x-X_i[i]),S[i], kernel_type ) 
    
    return fx

def gx(w,S,X_i,x, kernel_type='Gaussian'):
    """
    Return gradiant of Kernel Density value of a point accordung to following function
    
    g(x)= Σ_(i=1)^n w_i*∇K_S_i(x-X_i)
    """
    gx=np.zeros(shape=(1,len(x)))
    for i in range(len(w)):                                       
        gx=gx + w[i]* first_der_K_S(x-X_i[i],S[i], kernel_type ) 
    
    return gx

def Hx(w,S,X_i,x, kernel_type='Gaussian'):
    """
    Return Hessian of Kernel Density value of a point accordung to following function
    
    H(x)= Σ_(i=1)^n w_i*∇^2 (K_S_i(x-X_i))
    """  
    dim=len(x)
    Hx=np.zeros(shape=(dim,dim))
    for i in range(len(w)):
       Hx=Hx + w[i]* sec_der_K_S((x-X_i[i]),S[i], kernel_type) 
    
    return Hx

def ln_fx(w,S,X_i,x, kernel_type='Gaussian'):
    """"logarithm of Kernel density function fx(w,S,X_i,x, kernel_type='Gaussian')
    """
    return np.log(fx(w,S,X_i,x, kernel_type))

def ln_gx(w,S,X_i,x, kernel_type='Gaussian'):
    """logarithm of gradient of Kernel density function fx(w,S,X_i,x, kernel_type='Gaussian')
    ∇_f(x)=g(x)/f(x)
    """       
    return gx(w,S,X_i,x, kernel_type) / fx(w,S,X_i,x, kernel_type)

def ln_Hx (w,S,X_i,x, kernel_type='Gaussian'):
    """logarithm of hessian of  Kernel density function fx(w,S,X_i,x, kernel_type='Gaussian')
     ∇^T ∇_f(x)=Hx()/fx() -ln_gx* ln_gx^T
    """
    return Hx(w,S,X_i,x, kernel_type) / fx(w,S,X_i,x, kernel_type) - ln_gx(w,S,X_i,x, kernel_type).T*(ln_gx(w,S,X_i,x, kernel_type))
    

def estimateGrad(fun,x,delta):
    """ Given a real-valued function fun, estimate its gradient numerically. """
    d = len(x)
    grad = np.zeros(d)
    for i in range(d):
        e = np.zeros(d)
        e[i] = 1.0
        grad[i] = (fun(x+delta*np.transpose(np.mat(e))) - fun(x))/delta
    return grad


def estimateHes(gradFun,x,delta):
    """ Given a real-valued grad function, estimate its hessian numerically. """
    d = len(x)
    hes = np.zeros([d,d])
    for i in range(d):
        e = np.zeros(d)
        e[i] = 1.0
        hes[i] = (gradFun(x+delta*np.transpose(np.mat(e))) - gradFun(x))/delta
    return hes
 










 






