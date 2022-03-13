import numpy as np
from CustomerSupportMDPClass import CustomerSupportMDPClass
from vi_exp import vi_exp

mdp = CustomerSupportMDPClass();
# ----------------------------------  Do standard value iteration ---------------------
maxIter = 1e3;
tol = 1; #1e-5;
dis = 0.95; #dicont

[V_Exp,Pol_Exp,err_Exp] = vi_exp(mdp,tol,maxIter,dis);
print(Pol_Exp)
print(V_Exp)