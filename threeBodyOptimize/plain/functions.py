'''uses NumPy'''
import numpy as np
# From here Dario's functions
def GR2(r12,r2,r1,beta,Z,A,c200):
    y = A*np.exp(-Z*(r1+r2))*(1.0 + c200 *(r1**2 + r2**2))* \
    (1 + 2 * beta - np.exp(-beta *r12) )/(2 * beta)
    return y

def nabla1(r12,r2,r1,beta,Z,A,c200):
    y = -A*( (2*beta + 1)*np.exp(beta*r12) - 1) * \
    (r1*Z**2 * (c200 * (r1**2 + r2**2) + 1) - \
     2*Z*( c200 * (3*r1**2 + r2**2) + 1) + 6*c200 * r1)* \
    np.exp(Z*(-r1-r2)-beta*r12) /( 4 * beta * r1)
    return y

def nabla2(r12,r2,r1,beta,Z,A,c200):
    y = -A*( (2*beta + 1)*np.exp(beta*r12) - 1) * \
    (r2*Z**2 * (c200 * (r1**2 + r2**2) + 1) - \
     2*Z*( c200 * (3*r2**2 + r1**2) + 1) + 6*c200 * r2)* \
    np.exp(Z*(-r1-r2)-beta*r12) /( 4 * beta * r2)
    return y

def nabla12(r12,r2,r1,beta,Z,A,c200):
    y = A*(beta*r12 - 2)*(c200*(r1**2 + r2**2) + 1) * \
    np.exp(Z*(-r1-r2)-beta*r12) / (2*r12)
    return y

def d1d12(r12,r2,r1,beta,Z,A,c200):
    y = -A*1.0/2.0*( c200 *Z*(r1**2+r2**2) - 2.0 * c200 *r1 + Z) * \
    np.exp(Z*(-r1-r2) - beta*r12)
    return y  

def  t1(r12,r2,r1):
    rnum = r1**2 - r2**2 + r12**2
    rden = 2.0*r1*r12
    return rnum/rden

def d2d12(r12,r2,r1,beta,Z,A,c200):
    y = -A*1.0/2.0*( c200 *Z*(r1**2+r2**2) - 2.0 * c200 *r2 + Z) * \
    np.exp(Z*(-r1-r2) - beta*r12)
    return y  

def  t2(r12,r2,r1):
    rnum = r2**2 - r1**2 + r12**2
    rden = 2.0*r2*r12
    return rnum/rden

def H(r12,r2,r1,beta,Z,A,c200):
    y = nabla1(r12,r2,r1,beta,Z,A,c200) + \
    nabla2(r12,r2,r1,beta,Z,A,c200) + \
    nabla12(r12,r2,r1,beta,Z,A,c200) - \
    t1(r12,r2,r1)*d1d12(r12,r2,r1,beta,Z,A,c200) - \
    t2(r12,r2,r1)*d2d12(r12,r2,r1,beta,Z,A,c200) + \
    (1/r12 - Z/r1 - Z/r2)*GR2(r12,r2,r1,beta,Z,A,c200)
    return y


def Elocal(r12,r2,r1,beta,Z,A,c200):
    y = H(r12,r2,r1,beta,Z,A,c200)/GR2(r12,r2,r1,beta,Z,A,c200)
    return y    


