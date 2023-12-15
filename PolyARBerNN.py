# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:57:40 2020

@author: Wael
"""


from z3 import *
from yices import *
from scipy.optimize import minimize
from autograd import elementwise_grad as egrad
from autograd import jacobian
from autograd import grad
import itertools 
import numpy as np
import polytope as pc
import multiprocessing as mp
from multiprocessing import Pool, Process, Queue
from functools import partial
import time
import keras





class PolyInequalitySolver:

    # ========================================================
    #   Constructor
    # ========================================================
    def __init__(self, num_vars, boxber, pregion, orders):
        self.num_vars = num_vars
        #self.LipThres=LipThres
        #self.bounds = bounds
        self.boxber = boxber
        self.pregion = pregion
        
        self.orders = orders
        
        self.poly_inequality_coeffs  = []

        self.__SMT_region_bounds      = 0.01    # threshold below which we use SMT solver/CAD to compute the sign of the polynomial
        self.status=False
        

        

        

    

    # ========================================================
    #   Add A Polynomial Inequality
    # ========================================================
    def addPolyInequalityConstraint(self, poly):
        
        self.poly_inequality_coeffs.append(poly)
        
        
        
    

    # ========================================================
    #   Output s string in this form:'x0 x1...xn'
    # ========================================================
    
    def strVari(self,n):
        variables=''
        for i in range(0,n):
            variables=variables+'x'+str(i)+' '
        c= variables[:-1]  
        return c   
    
    # ========================================================
    #   Output s string in this form: '(* val xi)' 
    # ========================================================
    
    def strterm(self,val,i):
        valstr=str(val)
        istr=str(i)
        term='(*'+' '+valstr+' '+'x'+istr+')'
        return term
    
    # ========================================================
    #   Output s string in this form: '(* coef (^ xi i)....' 
    # ========================================================
    
    def strtermpoly(self,coef,pows):
        coefstr=str(coef)
        termpoly='(* '+coefstr
        
        for i in range(len(pows)):
            if pows[i] !=0:
                istr=str(i)
                powstr=str(pows[i])
                termpoly=termpoly+' '+'(^ '+'x'+istr+' '+powstr+')'
            
        termpoly=termpoly+')'    
        return termpoly
        
    # ========================================================
    #   1) Output list of fmla bounds string for yices2
    # ========================================================
    
    def fmlabounds1(self,pregion):
        fmlastrlist=[]
        A=pregion[0]['A']
        b=pregion[0]['b']
        
        for i in range(len(A[:,1])):
            fmlastr='(<=(+'
            strb=str(b[i])
            aux=''
            for j in range(self.num_vars):
                aux=aux+self.strterm(A[i,j],j)
                
            fmlastr=fmlastr+aux+')'+''+strb+')'    
            fmlastrlist.append(fmlastr)
            
        return fmlastrlist 
    
    # ========================================================
    #   2) Output list of fmla bounds string for yices2
    # ========================================================
    
    def fmlabounds2(self,box):

        
        fmlastr='(and'
        for i in range(self.num_vars):
            strb1=str(box[0][i][0])
            strb2=str(box[1][i][0])
             
            fmlastr=fmlastr+'( >= '+'x'+str(i)+' '+strb1+')'+''+'( <= '+'x'+str(i)+' '+strb2+')'   

            
        fmlastr=fmlastr+')'  
        return fmlastr 
    
    
    # ========================================================
    #   Output fmla poly string for yices2
    # ========================================================
    
    def fmlapoly(self,poly):
        
        fmlastrpoly='(<=(+'
        for monomial_counter in range(0,len(poly)):
            coeff = poly[monomial_counter]['coeff']
            vars  = poly[monomial_counter]['vars']
    
            
            pows=[]
            for var_counter in range(len(vars)):
                power = vars[var_counter]['power']
                pows.append(power)
            
            fmlastrpoly=fmlastrpoly+' '+ self.strtermpoly(coeff,pows)
        
        fmlastrpoly=fmlastrpoly+')'+' '+'0)'
        return fmlastrpoly 
    
    
    
    # ========================================================
    #   Function to output the string s in the poly structure
    #                    for our algorithm
    # ========================================================
    def polyconstr(self,s):
        varlist= symbols(self.strVari(self.num_vars))   
        
        numden=fraction(together(s))
        s= numden[0]* numden[1]    
        
        polycs=[]
        term={}
        poly=sympy.poly(s,varlist)
    
        polypowers=poly.monoms()
        polycoeffs=poly.coeffs()
        
        for j in range(len(polycoeffs)):
            varspows=[]
            for k in range(len(polypowers[j])):
                varspows.append({'power':polypowers[j][k]})
                
            term={'coeff':polycoeffs[j],'vars':varspows}
            polycs.append(term)   
                
    
        return polycs
    
        
    # ========================================================
    # Compute the approx Lipchtz constant L of multivar poly 
    # in region
    # ========================================================    
    def Lipchtz(self,poly,region,num_samples):
        all_list=[]
        for i in range(self.num_vars):
            X=np.linspace(region[i]['min'], region[i]['max'], num_samples, endpoint=True)
            all_list.append(X)
        all_coords=list(itertools.product(*all_list))
        all_coordsarray=[]
        for i in range(len(all_coords)):
            all_coordsarray.append(list(all_coords[i]))
        all_coordsarray=np.array(all_coordsarray)   
        
        poly_vals=[]
        for i in range(len(all_coordsarray)):
            poly_vals.append(self.evaluate_multivar_poly(poly,all_coordsarray[i]))
        poly_vals=np.array(poly_vals) 
        poly_vals_diff=np.diff(poly_vals)
        all_coordsarray_diff=np.diff(all_coordsarray,axis=0)
        all_coordsarray_diff=np.linalg.norm(all_coordsarray_diff,axis=1)
        L=max(abs(poly_vals_diff)/all_coordsarray_diff)
        return L
    
    # ========================================================
    # Partition region into subregions around the components 
    # that have higher rate change threshold
    # ========================================================  
    def Partition_ratechange(self,poly,polype,num_samples):
        
        ambiguous_regions=[]
#        Areg=pregion[0]['A']
#        breg=pregion[0]['b']
#        polype=pc.Polytope(Areg, breg) 
        boundingbox=pc.bounding_box(polype)
        
        all_list=[]
        for i in range(self.num_vars):
            X=np.linspace(boundingbox[0][i][0], boundingbox[1][i][0], num_samples, endpoint=True)
            all_list.append(X)
            
        sampldist=all_list[0][1]- all_list[0][0]   
        
        all_coords=list(itertools.product(*all_list))
        all_coordsarray=[]
        for i in range(len(all_coords)):
            all_coordsarray.append(list(all_coords[i]))
        all_coordsarray=np.array(all_coordsarray)  
        
        ratechange=[]
        for i in range(len(all_coordsarray)):
            ratechange.append(np.linalg.norm(self.Gradient(poly,all_coordsarray[i])))
            
        res = [] 
        for idx in range(0, len(ratechange)) : 
            if ratechange[idx] > 50000000: 
                res.append(idx) 
                
        if len(res)==0:
            return 0, ambiguous_regions

        highratechangev=all_coordsarray[res,:]

        ps=[]
        for i in range(len(highratechangev)):
            box=[]
            for j in range(self.num_vars):
                box.append([highratechangev[i][j]-sampldist,highratechangev[i][j]+sampldist])    
            box=np.array(box)
            p=pc.box2poly(box)
            p=p.intersect(polype)
            ps.append(p)
            
        psreg=pc.Region(ps)  
        pambig=(polype.diff(psreg)).union(psreg) 

        for polytope in pambig:
            ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])  
        
        return 1,ambiguous_regions
        #return 1,ratechange
        
        
        
        
        
        
    
    # ========================================================
    # Partition polype to 2 polypes along the long dimension 
    # ========================================================    
    def Part_polype(self,polype):
        box=pc.bounding_box(polype)
        boxn=np.append(box[0], box[1], axis=1)
        indexmax=np.argmax(box[1]-box[0])
        mid=0.5*(boxn[indexmax][0]+boxn[indexmax][1])
        
        box1=boxn
        box2=boxn
        box1=np.delete(box1, indexmax, axis=0)
        box1=np.insert(box1, indexmax, np.array([boxn[indexmax][0],mid]), axis=0)
        
        box2=np.delete(box2, indexmax, axis=0)
        box2=np.insert(box2, indexmax, np.array([mid,boxn[indexmax][1]]), axis=0)
        
        p1=pc.box2poly(box1)
        p2=pc.box2poly(box2)
        
        p1=polype.intersect(p1)
        p2=polype.intersect(p2)
        
        return p1,p2

    # ========================================================
    #   Compute the Hessian Matrix of multivar poly at point x
    # ========================================================
    def Hessian(self,poly,x):
        
        def multivar_poly(x):
            result = 0
            for monomial_counter in range(0,len(poly)):
                coeff = poly[monomial_counter]['coeff']
                vars  = poly[monomial_counter]['vars']
                product = coeff
                for var_counter in range(len(vars)):
                    power = vars[var_counter]['power']
                    var   = x[var_counter]  
                    product = product * (var**power)
                result = result + product
            return result
       
        H_f = jacobian(egrad(multivar_poly)) 
        return H_f(x)
    
    # ========================================================
    #   Compute the Gradient Vector of multivar poly at point x
    # ========================================================
    def Gradient(self,poly,x):
    
        def multivar_poly(x):
            result = 0
            for monomial_counter in range(0,len(poly)):
                coeff = poly[monomial_counter]['coeff']
                vars  = poly[monomial_counter]['vars']
                product = coeff
                for var_counter in range(len(vars)):
                    power = vars[var_counter]['power']
                    var   = x[var_counter]  
                    product = product * (var**power)
                result = result + product
            return result    
    
        grad_f = grad(multivar_poly)  
        return grad_f(x)
    
    
    # ========================================================
    #   Evaluate the multivar poly at point x
    # ========================================================
    
    def evaluate_multivar_poly(self,poly, x):
        result = 0
        for monomial_counter in range(0,len(poly)):
            coeff = poly[monomial_counter]['coeff']
            vars  = poly[monomial_counter]['vars']
            product = coeff
            for var_counter in range(len(vars)):
                power = vars[var_counter]['power']
                var   = x[var_counter]  
                product = product * (var**power)
            result = result + product
        return result
    
    
    # ========================================================
    #   Check if a matrix M is positive semidefinite or not
    # ========================================================
    def is_pos_sem_def(self,M):
        return np.all(np.linalg.eigvals(M) >= 0)
    
    # ========================================================
    #   Check if a matrix M is positive definite or not
    # ========================================================
    def is_pos_def(self,M):
        return np.all(np.linalg.eigvals(M) > 1e-8)
    
    # ========================================================
    #   Check if a matrix M is negative definite or not
    # ========================================================
    def is_neg_def(self,M):
        return np.all(np.linalg.eigvals(M) < 0)
    
    # ========================================================
    #   Number of positive eigenvalues: Lambda_i>0
    # ========================================================
    def num_pos_eig(self,M):
        w=np.linalg.eigvals(M)
        return np.sum(w > 1e-2)
    
        
    # ========================================================
    #   Compute the upper bound of 1st ord Remin of Taylor app
    # ========================================================
    def remainder1cst(self,poly,pregion,mid_point,Gradi):
        
        cons=[{'type':'ineq','fun':lambda x:pregion[0]['b']-(pregion[0]['A']).dot(x)}]
        
        objectiveFunction = lambda x:-abs(self.evaluate_multivar_poly(poly,x)-(self.evaluate_multivar_poly(poly,mid_point)+Gradi.dot((x-mid_point))))
        res = minimize(objectiveFunction, np.ones(self.num_vars)/(self.num_vars), constraints=cons, options={'disp': False})
        # np.random.uniform(low=lowvals, high=highvals, size=self.num_vars)
        regneg=res.fun
        # print(regneg)
#        ccc=abs(self.evaluate_multivar_poly(poly,regneg)-(self.evaluate_multivar_poly(poly,mid_point)+Gradi.dot((regneg-mid_point))))
#        print(ccc)
                    
        stat=res.status
        # print(stat)
        if stat==0:
            return -regneg
        else:
            b=[]
            return b
    
    
    # ========================================================
    #   Compute the upper bound of 2nd ord Remin of Taylor app
    # ========================================================
    
    
    def remainder2cst(self,poly,pregion,mid_point,Gradi,Hess):
        
        cons=[{'type':'ineq','fun':lambda x:pregion[0]['b']-(pregion[0]['A']).dot(x)}]
        
        objectiveFunction = lambda x:-abs(self.evaluate_multivar_poly(poly,x)-(self.evaluate_multivar_poly(poly,mid_point)+Gradi.dot((x-mid_point))+0.5*(x-mid_point).dot(Hess.dot((x-mid_point)))))
        res = minimize(objectiveFunction, np.ones(self.num_vars)/(self.num_vars), constraints=cons, options={'disp': False})
        # np.random.uniform(low=lowvals, high=highvals, size=self.num_vars)
        regneg=res.fun
        #print( regneg)
#        ccc=abs(self.evaluate_multivar_poly(poly,regneg)-(self.evaluate_multivar_poly(poly,mid_point)+Gradi.dot((regneg-mid_point))+0.5*(regneg-mid_point).dot(Hess.dot((regneg-mid_point)))))
#        print(ccc)
           
        stat=res.status
        # print(stat)
        if stat==0:
            return -regneg
        else:
            b=[]
            return b
    
        

 
    
    
    # ========================================================
    # Output:the rectangle incribed in polytope: A,b
    # ========================================================     
    def RecinPolytope(self,A,b):
        Ap=A.clip(0)
        Am=(-A).clip(0)
        cons=[{'type':'ineq','fun':lambda x:b-(Ap.dot(x[0:self.num_vars]))+(Am.dot(x[self.num_vars:]))},{'type':'ineq','fun':lambda x:x[0:self.num_vars]-x[self.num_vars:]-0.001}]
        
    
         
        objectiveFunction = lambda x: -np.prod(x[0:self.num_vars]-x[self.num_vars:])
        #objectiveFunction = lambda x: -np.sum(np.log(x[0:self.num_vars]-x[self.num_vars:3]))
        #objectiveFunction = lambda x: -(x[0]-x[2])*(x[1]-x[3])
        res = minimize(objectiveFunction, np.ones(2*(self.num_vars))/(2*(self.num_vars)), constraints=cons, options={'disp': False})
        #
        regneg=res.x
                    
        stat=res.status
        #print(res)
        if stat==0:
            b=(np.array([regneg[self.num_vars:]]).T,np.array([regneg[0:self.num_vars]]).T)
            return b
        else:
            b=[]
            return b     
    
    
 
   
        
    # ========================================================
    # Output:Vertex v of under-approx polytope tangent to tem (one sheet case)
    # ========================================================     
    def Ver_tang_one_sheet(self,poly,pregion,mid_point,Hess,Gradi,Rem2,Tem,sign):

        if sign=='N':    
            cons=[{'type':'ineq','fun':lambda x:-(self.evaluate_multivar_poly(poly,mid_point)+Gradi.dot((x-mid_point))+0.5*(x-mid_point).dot(Hess.dot((x-mid_point)))+Rem2+0.01)},{'type':'ineq','fun':lambda x:pregion[0]['b']-(pregion[0]['A']).dot(x)}]
    #        cons=[{'type':'ineq','fun':lambda x:-(self.evaluate_multivar_poly(poly,x))}]
            
    
        
            objectiveFunction = lambda x: -x.dot(Tem)
            res = minimize(objectiveFunction, np.ones(self.num_vars)/(self.num_vars), constraints=cons, options={'disp': False})
            # np.random.uniform(low=lowvals, high=highvals, size=self.num_vars)
            regneg=res.x
                        
            stat=res.status
            # print(res)
            # print(stat)
            if stat==0:
                return regneg
            else:
                b=[]
                return b  
            
        else:   
            cons=[{'type':'ineq','fun':lambda x:(self.evaluate_multivar_poly(poly,mid_point)+Gradi.dot((x-mid_point))+0.5*(x-mid_point).dot(Hess.dot((x-mid_point)))-Rem2)},{'type':'ineq','fun':lambda x:-pregion[0]['b']+(pregion[0]['A']).dot(x)}]
    #        cons=[{'type':'ineq','fun':lambda x:-(self.evaluate_multivar_poly(poly,x))}]
            
    
        
            objectiveFunction = lambda x: -x.dot(Tem)
            res = minimize(objectiveFunction, np.ones(self.num_vars)/(self.num_vars), constraints=cons, options={'disp': False})
            # np.random.uniform(low=lowvals, high=highvals, size=self.num_vars)
            regneg=res.x
                        
            stat=res.status
            # print(stat)
            if stat==0:
                return regneg
            else:
                b=[]
                return b 
    # ========================================================
    # Output:Vertex v of under-approx polytope tangent to tem
    # (two sheets case): The left side of the hyperplane:
    #                   Ax<=b
    # ========================================================     
    def Ver_tang_two_sheet(self,poly,pregion,mid_point,Hess,Gradi,Rem2,Tem,A,b,sign):

        if sign=='N':    
            cons=[{'type':'ineq','fun':lambda x:-(self.evaluate_multivar_poly(poly,mid_point)+Gradi.dot((x-mid_point))+0.5*(x-mid_point).dot(Hess.dot((x-mid_point)))+Rem2)},{'type':'ineq','fun':lambda x:b-A.dot(x)},{'type':'ineq','fun':lambda x:pregion[0]['b']-(pregion[0]['A']).dot(x)}]
    #        cons=[{'type':'ineq','fun':lambda x:-(self.evaluate_multivar_poly(poly,x))}]
            
        
            objectiveFunction = lambda x: -x.dot(Tem)
            res = minimize(objectiveFunction, np.ones(self.num_vars)/(self.num_vars), constraints=cons, options={'disp': False})
            # np.random.uniform(low=lowvals, high=highvals, size=self.num_vars)
            regneg=res.x
                        
            stat=res.status
            # print(stat)
            if stat==0:
                return regneg
            else:
                b=[]
                return b   
        else:
            cons=[{'type':'ineq','fun':lambda x:(self.evaluate_multivar_poly(poly,mid_point)+Gradi.dot((x-mid_point))+0.5*(x-mid_point).dot(Hess.dot((x-mid_point)))-Rem2)},{'type':'ineq','fun':lambda x:-b+A.dot(x)},{'type':'ineq','fun':lambda x:pregion[0]['b']-(pregion[0]['A']).dot(x)}]
    #        cons=[{'type':'ineq','fun':lambda x:-(self.evaluate_multivar_poly(poly,x))}]
            
        
            objectiveFunction = lambda x: -x.dot(Tem)
            res = minimize(objectiveFunction, np.ones(self.num_vars)/(self.num_vars), constraints=cons, options={'disp': False})
            # np.random.uniform(low=lowvals, high=highvals, size=self.num_vars)
            regneg=res.x
                        
            stat=res.status
            # print(stat)
            if stat==0:
                return regneg
            else:
                b=[]
                return b
            
    
    def quitfc(self,result):
        if result:
            self.p.terminate()     
            
            
            
            
    
    # ========================================================
    #   Compute the binomial of n choose k 
    # ========================================================
    def binom(self, n, k):
        return math.factorial(n) // math.factorial(k) // math.factorial(n - k)
    
    
    
        # ========================================================
      # compute min bernstein coeffs of the univariate monomial 
      # x_{i}^{k} in the interval [xmin_i, xmax_i], 
      # where k is the power and l_i is the order
      # inputs:
      # k: the power
      # l_i: the order 
      # interval = [xmin_i, xmax_i]
      # output:
      # bern, list of the bernstein coefficients
      # ========================================================
      
    def min_univ_monom_bernst(self, k, l_i, interval):
          
        if k == 0:
            sum = 1
            return sum
        
        else:
            i = 0
            
            if k == l_i:
                sum = interval[0]**(k - i) * interval[1]**i
                return sum
            
            else:
            
                sum = 0
                min_i_k = min(i, k)
                for j in range(min_i_k + 1):
                    
                    sum = sum + (self.binom(i, j)/self.binom(l_i, j)) * (interval[1] - interval[0])**j * self.binom(k, j) * interval[0]**(k - j)
                
            return sum 
      
      # ========================================================
      # compute max bernstein coeffs of the univariate monomial 
      # x_{i}^{k} in the interval [xmin_i, xmax_i], 
      # where k is the power and l_i is the order
      # inputs:
      # k: the power
      # l_i: the order 
      # interval = [xmin_i, xmax_i]
      # output:
      # bern, list of the bernstein coefficients
      # ========================================================
      
    def max_univ_monom_bernst(self, k, l_i, interval):
          
        if k == 0:
            sum = 1
            return sum
        
        else:
            
            i = l_i
            
            if k == l_i:
                sum = interval[0]**(k - i) * interval[1]**i
                return sum
            
            else:
                sum = 0
                min_i_k = min(i, k)
                for j in range(min_i_k + 1):
                    
                    sum = sum + (self.binom(i, j)/self.binom(l_i, j)) * (interval[1] - interval[0])**j * self.binom(k, j) * interval[0]**(k - j)
                    # print(sum)
                
            return sum  
      
              
      
      # ========================================================
      # compute min bernstein coeffs of the multivariate polynomial
      # poly in the box using implicit berns representation:
      # inputs:
      # poly: polynomial into consideration    
      # box: region into consideration 
      # Lpoly: orders of the polynomial    
      # output:
      # bern: matrix that represents the bernstein coefficients
      # of poly over box    
      # ========================================================
      
    def min_poly_implicit_bernst(self, poly, box, Lpoly):
          
        min_bern = 0
        
        for term in poly:
            
            min_bern_term = 1
            
            a = term['coeff']
            
            i = 0
            
            for x in term['vars']:
                
                if a > 0:
                    
                    min_res = self.min_univ_monom_bernst(x['power'], Lpoly[i], box[i])
                    min_bern_term = min_bern_term * (min_res)
                    
                elif (a < 0):    
                    
                    min_res = self.max_univ_monom_bernst(x['power'], Lpoly[i], box[i])
                    min_bern_term = min_bern_term * (min_res)
                
                                        
                i = i + 1
               
               
            min_bern  = min_bern + a * min_bern_term
            
            
        return min_bern
      
      
      
      
      
      
      # ========================================================
      # compute max bernstein coeffs of the multivariate polynomial
      # poly in the box using implicit berns representation:
      # inputs:
      # poly: polynomial into consideration    
      # box: region into consideration 
      # Lpoly: orders of the polynomial    
      # output:
      # bern: matrix that represents the bernstein coefficients
      # of poly over box    
      # ========================================================
      
    def max_poly_implicit_bernst(self, poly, box, Lpoly):
    
          
        max_bern = 0
        
        for term in poly:
            
            max_bern_term = 1
            
            a = term['coeff']
        
            i = 0
            
            for x in term['vars']:
                
        
                if a > 0:
                    
                    max_res = self.max_univ_monom_bernst(x['power'], Lpoly[i], box[i])
                    max_bern_term = max_bern_term * (max_res)
                    
                    
                elif (a < 0):
                    
                    max_res = self.min_univ_monom_bernst(x['power'], Lpoly[i], box[i])
                    max_bern_term = max_bern_term * (max_res)
                    
                
                    
                    
                i = i + 1
               
               
            max_bern  = max_bern + a * max_bern_term
            
            
        return max_bern
    
    
    def UNSAT_Remov_Berns(self, poly_list):
    
        i = 0
        # for poly, order in zip([poly_list[0]], [self.orders[0]]):
        for poly, order in zip(poly_list[-2:], self.orders[-2:]):
        # while (len(poly_list) != 0) :    
            
            
            # print('i', i)
            # print(len(poly_list))
            
            min_bern = self.min_poly_implicit_bernst(poly, self.boxber, order)
            
            max_bern = self.max_poly_implicit_bernst(poly, self.boxber,  order)
            
            # print('max_bern', max(min_bern, max_bern))
            
            # print('min_bern', min(min_bern, max_bern))
            
            if min_bern > 0:
                
                return 'UNSAT'
            
            
            
            if max_bern  <= 0:
                
                poly_list.remove(poly)
                self.orders.remove(order)
                
            # else:    
                
            #     i = i + 1    
         
                
        return 'SAT_UNSAT'  
                
            
        
    # ========================================================
    #   The main solver that solve the multivar constraints
    # ========================================================
    def solve(self):
        if self.poly_inequality_coeffs == []:
            print('ERROR: At least one polynomial constraint is needed')
            return self.bounds[0]['min']
        

        poly_list=list(range(0,len(self.poly_inequality_coeffs)))
        
        result = self.UNSAT_Remov_Berns(self.poly_inequality_coeffs)

        if (result == 'UNSAT') or (len(poly_list) == 0):
            return 'UNSAT'


        
        negative_regions=self.pregion
        aux=[]
        aux2=[]
        
        while poly_list:
            aux=negative_regions  
            poly_index=poly_list.pop(0)
            
            
            
            # compute min/max Bernstein coeffs and gradients coefficients
            minb, maxb = self.min_max_Ber(aux, poly_index)
            minb_grad, maxb_grad   = self.min_max_Ber_grad(aux, poly_index)
            

            
            # load the trained NN.
            NN_model = keras.models.load_model("model")
            
            # predict the right action
            input_NN = np.array([[minb, maxb, minb_grad, maxb_grad]])
            actions = NN_model(input_NN)
            
            # print(actions)
            
            # compute the index of the right action, could be 0, 1, or 3
            max_action =  max(list(actions[0]))
            action_index =list(actions[0]).index(max_action)
            
            if action_index == 0:
                ambiguous_regions,negative_regions = self.iterat(aux, poly_index,'N')
            elif action_index == 1:     
                ambiguous_regions,positive_regions = self.iterat(aux, poly_index,'P')
            else:
                ambiguous_region1, ambiguous_region2 = self.split(aux)
                
                ambiguous_regions = ambiguous_region1 + ambiguous_region2
                
            aux2.insert(0,ambiguous_regions)
            if not negative_regions:
                for i in range(len(aux2)):
                    aux3=aux2.pop(0)
                    for ccc in range(len(aux3)):
                        regionf=aux3.pop(0)
                        Areg=regionf[0]['A']
                        breg=regionf[0]['b']
                        polype=pc.Polytope(Areg, breg) 
                        
                        boxxpolype=pc.bounding_box(polype)
                        boxxpolype=np.append(boxxpolype[0], boxxpolype[1], axis=1)
                        polype=pc.box2poly(boxxpolype)
                                         
                        ###########################################################################
                        p1,p2=self.Part_polype(polype)
                        plist=[[p1,p2]]
                        kk=0
                        while kk<1:
                            plistsub=[]
                            for j in range(len(plist[kk])):
                                p1,p2=self.Part_polype(plist[kk][j])
                                plistsub=plistsub+[p1,p2]
                            plist.append(plistsub)   
                            kk=kk+1

                            
                        
                              
                        ##############################2)Parallel###################################   
                        iterable=[]
                        for kkk in range(len(plist[0])):   
                            boxx=pc.bounding_box(plist[0][kkk])   
                            iterable.append(boxx)

                        procs=[]
                        q=Queue()
                        for box in iterable:
                                
                            proc= Process(target=self.Yicesmany_multivars, args=(self.poly_inequality_coeffs,box,q))  
                            # proc= Process(target=self.solveZ3_many_multivars, args=(self.poly_inequality_coeffs,box,q)) 
                            proc.start()
                            procs.append(proc)
                            
                            
                        is_done = True
                        counter=0
                        while is_done:
                              time.sleep(0.01)
                              for process in procs:
                                  if (not process.is_alive()):
                                      if (not q.empty()):
                                          is_done = False
                                          self.status=True
                                          break
                                      else:
                                          counter=counter+1          
                              if counter==len(procs):
                                  break 
                              
                              counter=0

                                        
                                            
                        if self.status:                
                            for process in procs:
                                process.terminate()
                            return 'SAT', q.get()
                       
                        else:
                            for process in procs:
                                process.terminate()


                                
 
                return 'UNSAT'
                    
            aux=[]

        if negative_regions :
            if (len(negative_regions)==0):
                region = negative_regions
                polytope = pc.Polytope(region[0]['A'], region[0]['b'])
                r,sol=pc.cheby_ball(polytope)
            else:
                region = negative_regions[0]
                polytope = pc.Polytope(region[0]['A'], region[0]['b'])
                r,sol=pc.cheby_ball(polytope)
            return 'SAT'
        
        else:   
            return 'UNSAT' 
        
        
        
        
    # ========================================================
    # Compute the gradient of a polynomial poly 
    # ========================================================    
        
    def grad(self, poly):
        
        gradient = []
        
        occ_pow_0 = 0
        for i in range(self.num_vars):
            
            grad_poly_i = poly
            
            for monomial_counter in range(0,len(grad_poly_i)):
                coeff = grad_poly_i[monomial_counter]['coeff']
                vars  = grad_poly_i[monomial_counter]['vars']
                product = coeff
                
                
                for var_counter in range(len(vars)):
                    power = vars[var_counter]['power']
                    var   = x[var_counter]  
                    product = product * (var**power)
                    
                power = vars[i]['power']
                
                if power == 0:
                    
                    occ_pow_0 = occ_pow_0 + 1
                    vars[var_counter]['power'] == power
                    
                else:
                    
                    vars[var_counter]['power'] == power - 1
                    
                    
            gradient.append(grad_poly_i)       
                    
                    
        if occ_pow_0 == self.num_vars:
            
            for i in range(self.num_vars):
                
                gradient[i] = gradient[i][:-1]
                
        return gradient            
                    
                    
                    

            
        
        
        
        
    # ========================================================
    #   Compute the minBer and maxBer of the polytopes in 
    #    the list_regions
    # ======================================================== 
    def min_max_Ber(self,list_regions, poly_index):  
        
        
        minbaux = []
        maxbaux = []
        
        poly = self.poly_inequality_coeffs[poly_index]
        orders = self.orders[poly_index]
        
        if len(list_regions) != 0:
            for i in range(len(list_regions)):
                
                
                polype = pc.Polytope(list_regions[i][0]['A'], list_regions[i][0]['b']) 
                
                box=pc.bounding_box(polype)
                boxn=np.append(box[0], box[1], axis=1)
                
    
                
                minb = min(self.min_poly_implicit_bernst(poly, boxn, orders), self.max_poly_implicit_bernst(poly, boxn, orders))
                minbaux.append(minb)
                
    
                    
                
                
                
                maxb = max(self.min_poly_implicit_bernst(poly, boxn, orders), self.max_poly_implicit_bernst(poly, boxn, orders))
                maxbaux.append(maxb)
    
                
                
            
            # self.minBer.append(min(minbaux))  
            # self.maxBer.append(max(maxbaux))
            
            return min(minbaux), max(maxbaux)
        
        
    # ========================================================
    #   Compute the minBer and maxBer of the gradient of 
    #    polytopes in the list_regions
    # ======================================================== 
    def min_max_Ber_grad(self,list_regions, poly_index):  
        
        minb_list = []
        
        maxb_list = []
        
        poly = self.poly_inequality_coeffs[poly_index]
        
        orders = self.orders[poly_index].copy()
        
        polys_grads = self.grad_poly(poly, self.num_vars)
        
        if len(list_regions) != 0:
            for i in range(len(list_regions)):
                
                
                
                
                
                polype = pc.Polytope(list_regions[i][0]['A'], list_regions[i][0]['b']) 
                
                box=pc.bounding_box(polype)
                boxn=np.append(box[0], box[1], axis=1)
                
                for j in range(self.num_vars):
                    
                    orders_poly_grad = orders
                    orders_poly_grad[j] = orders_poly_grad[j] - 1
                    minb_reg = []
                    maxb_reg = []
                
                    minb = min(self.min_poly_implicit_bernst(polys_grads[j], boxn, orders_poly_grad), self.max_poly_implicit_bernst(polys_grads[j], boxn, orders_poly_grad))
                    minb_reg.append(minb)
                    
                    
                    maxb = max(self.min_poly_implicit_bernst(polys_grads[j], boxn, orders_poly_grad), self.max_poly_implicit_bernst(polys_grads[j], boxn, orders_poly_grad))
                    maxb_reg.append(maxb)
                    
    
                minb_list.append(minb_reg)
                maxb_list.append(maxb_reg)
                
            
            # self.minBer.append(min(minbaux))  
            # self.maxBer.append(max(maxbaux))
                # print(min(minbaux_1 + minbaux_2), max(maxbaux_1 + maxbaux_2))
            
        minb = min(minb_reg)
        maxb = max(maxb_reg)
        return minb, maxb      
  
    
    

        
    # ========================================================
    # check if there is a constant in the polynomial poly
    # ========================================================  
    
    def Exist_constant(self, poly, n):
        
        poly_copy = poly.copy()
        for monomial_counter in range(0,len(poly_copy)):
            
                vars  = poly_copy[monomial_counter]['vars']
                
                
                occ_pow_0 = 0
                
                for j in range(n):
                    
                    po = vars[j]['power']
    
                    
                    if po == 0:
                        
                        occ_pow_0 = occ_pow_0  + 1
                        
                if (occ_pow_0 == n):
                    
                    return 1
                
    
        
        return 0
    
    
    
        
    
    # ========================================================
    # Compute the list of powers of the i^{th} variable in gradient poly   
    # ======================================================== 
    def list_pows(self, poly, i):
    
        pows = []
    
        for monomial_counter in range(0,len(poly)):
            vars  = poly[monomial_counter]['vars']
            
            power = vars[i]['power'] 
            
            if power !=0:
    
                pows.append(power - 1)
                
            else:
                pows.append(power)
            
        return pows
        
    
    
         
    
    
    # ========================================================
    # Compute the gradient of a polynomial poly 
    # ========================================================    
        
    def grad_poly(self, poly, n):
        
    
        res = self.Exist_constant(poly, n)
        
        if res == 1:
            poly.pop()
    
        gradient = []
        for i in range(n):
            
            pows = self.list_pows(poly, i)
            
            poly_grad=[]
            term={}            
            for j in range(len(poly)):
                varspows=[]
                for k in range(n):
                    if k == i:
                        varspows.append({'power':pows[j]})
                        
                    else:
                        vars  = poly[j]['vars']
                        varspows.append({'power':vars[k]['power']})
                    
                term={'coeff':poly[j]['coeff'],'vars':varspows}
                poly_grad.append(term)  
                
            gradient.append(poly_grad)        
    
                
        return gradient        
        
        
    #========================================================
    #   Construct the constraints for the Quadratic Programm
    #========================================================      
        
    def MultVarCons(self,poly,pregion,point,x):
        Hess=self.Hessian(poly,point)
        Gradi=self.Gradient(poly,point)
        if self.is_pos_sem_def(Hess):# Hessian is sem def pos: Keep the poly
            remainder2=self.remainder2cst(poly,pregion,point,Gradi,Hess)
            out=self.evaluate_multivar_poly(poly,point)+Gradi.dot((x-point))+0.5*(x-point).dot(Hess.dot((x-point)))+remainder2
            return out
        else: # Hessian is not sem def pos: Keep the poly: Taylor overapprox 1st order
            remainder1=self.remainder1cst(poly,pregion,point,Gradi)
            out=self.evaluate_multivar_poly(poly,point)+Gradi.dot((x-point))+remainder1
            return out    
    
        
        
    
 
    # ========================================================
    #   Applying Multivar Quad Prog to find feasible point
    # ======================================================== 

    def Quad_Prog_mult_var(self,pregion,poly_list):
        
        # Transform pregion into polytope format
        Areg=pregion[0]['A']
        breg=pregion[0]['b']
        polype=pc.Polytope(Areg, breg) 
        
        # Compute the middle point in the polytope region
        rb,mid_point=pc.cheby_ball(polype)
#        print('mid_point')
#        print(mid_point)
        
        cons=[{'type':'ineq','fun':lambda x, poly=poly:-self.MultVarCons(self.poly_inequality_coeffs[poly],pregion,mid_point,x)-0.000001} for poly in poly_list]
        cons=cons+[{'type':'ineq','fun':lambda x:pregion[0]['b']-(pregion[0]['A']).dot(x)}]
        
#        cons=[{'type':'ineq','fun':lambda x:-self.MultVarCons(self.poly_inequality_coeffs[0],pregion,mid_point,x)},{'type':'ineq','fun':lambda x:-self.MultVarCons(self.poly_inequality_coeffs[1],pregion,mid_point,x)},{'type':'ineq','fun':lambda x:-self.MultVarCons(self.poly_inequality_coeffs[2],pregion,mid_point,x)},{'type':'ineq','fun':lambda x:-self.MultVarCons(self.poly_inequality_coeffs[3],pregion,mid_point,x)},{'type':'ineq','fun':lambda x:-self.MultVarCons(self.poly_inequality_coeffs[4],pregion,mid_point,x)},{'type':'ineq','fun':lambda x:-self.MultVarCons(self.poly_inequality_coeffs[5],pregion,mid_point,x)}]
        
#        cons=[{'type':'ineq','fun':lambda x, poly=poly:-self.evaluate_multivar_poly(self.poly_inequality_coeffs[poly],x)} for poly in poly_list]
#        cons=cons+[{'type':'ineq','fun':lambda x:pregion.b-(pregion.A).dot(x)}]

    


    
        objectiveFunction = lambda x: 0  
        res = minimize(objectiveFunction, np.ones(self.num_vars)/(self.num_vars), method='SLSQP', constraints=cons, options={'disp': False})
        # # np.random.uniform(low=lowvals, high=highvals, size=self.num_vars)
        regneg=res.x
        
       
        stat=res.status
            # print(stat)
        if stat==0:
            return regneg
        else:
            b=[]
            return b      
        
        
        
        
    def split(self, ambig_reg):
        
        Areg=ambig_reg[0][0]['A']
        breg=ambig_reg[0][0]['b']
        polype=pc.Polytope(Areg, breg) 
        
        p1,p2=self.Part_polype(polype)
        
        ambig_reg_1 = [[{'A':p1.A,'b':p1.b}]]
        
        ambig_reg_2 = [[{'A':p2.A,'b':p2.b}]]
        
        return ambig_reg_1, ambig_reg_2
        
    
    
        
    # ========================================================
    #   Iterative alg to partition Ambig reg until gets small
    # ======================================================== 
    def iterat(self,aux_ambigreg,poly_counter,sign):

        aux_neg_pos_reg=[] 
         
        while aux_ambigreg:
              region = aux_ambigreg.pop(0)
              polype=pc.Polytope(region[0]['A'], region[0]['b']) 
              if polype.volume <1000**5/0.1:
                  aux_ambigreg.append(region)


                  return  aux_ambigreg, aux_neg_pos_reg
             
              else:
                 
                 
                  if sign=='N':
                      ambiguous_regions, negative_regions=self.Npartition_inequality_regions(poly_counter, region)
                     
                      aux_ambigreg=aux_ambigreg+ambiguous_regions
                      aux_neg_pos_reg=aux_neg_pos_reg+negative_regions
                  else:
                      ambiguous_regions, positive_regions=self.Ppartition_inequality_regions(poly_counter, region)
                      aux_ambigreg=aux_ambigreg+ambiguous_regions
                      aux_neg_pos_reg=aux_neg_pos_reg+positive_regions


        return  aux_ambigreg, aux_neg_pos_reg         
     
 
            
        
            
        
    # ========================================================
    # Partition the region (Neg+Ambig) based on the polynomial sign
    # ========================================================  
    def Npartition_inequality_regions(self,inequality_index, pregion): 
        poly=self.poly_inequality_coeffs[inequality_index]
        
        # Initialization of the neg, ambig, regions
        negative_regions        = []
        ambiguous_regions       = [] 
        # Transform pregion into polytope format
        Areg=pregion[0]['A']
        breg=pregion[0]['b']
        polype=pc.Polytope(Areg, breg) 
                
        # Compute the middle point in the polytope region
        rb,mid_point=pc.cheby_ball(polype)
        # Compute the hessian Matrix of Poly at mid_point
        Hess=self.Hessian(poly,mid_point)
        # print('cvcvcvcvcv1')
    
        # Compute the gradient vector of Poly at mid_point
        Gradi=self.Gradient(poly,mid_point)
        
        # Compute the remainder for the 2nd order Taylor overapproximation
        Rem2=self.remainder2cst(poly,pregion,mid_point,Gradi,Hess)
        
        if self.num_pos_eig(Hess)==1: # There is two sheets
            # print('2sheets')
            # Compute the hyperplane (As,bs) that will separate the two sheets
              
            #  Compute the center of the hyperbola
            coefcen=Hess.dot(mid_point)-Gradi
            #print(Hess)
            cen=np.linalg.solve(Hess,coefcen)
            # Compute the principal axis As
            eigval,eigvec=np.linalg.eig(Hess)
            # Compute the index of the eigenvalue > 0
            indexaux=np.nonzero(eigval> 0)
            index=indexaux[0][0]
            # Compute As and bs
            As=eigvec[index,:]
            bs=As.dot(cen)
             
            # Compute 2n vertices (2n faces) of the two polytope that under-approximate the negative regions if they exist      
            # Construct 2n template vectors 
            Tem1=np.identity(self.num_vars)
            Tem2=-np.identity(self.num_vars)
            Tem=np.insert(Tem1,self.num_vars,Tem2,axis=0)
        
            # Compute the vertices of the two polytopes  
            vertices1=[]
            vertices2=[]
            for i in range((self.num_vars)+1):

                v1=self.Ver_tang_two_sheet(poly,pregion,mid_point,Hess,Gradi,Rem2,Tem[i,:],As,bs,'N')
                v2=self.Ver_tang_two_sheet(poly,pregion,mid_point,Hess,Gradi,Rem2,Tem[i,:],-As,-bs,'N')
                v1=list(v1)
                v2=list(v2)
            
            
                # if the vertices exist==> Neg Reg exist==> Save it
                # print(v1,v2)
                if (len(v1)!=0) and (len(v2)!=0):
                    vertices1.append(v1)
                    vertices2.append(v2)
                elif (len(v1)!=0):  
                    vertices1.append(v1)
                elif (len(v2)!=0):  
                    vertices2.append(v2)    
                else: # if the vertices  does not exist==> Neg Reg does not exist==> Partition Ambiguous Region
                    p1,p2=self.Part_polype(polype)
                    pambig=p1.union(p2)

                    if len(pambig)==0:
                        if pambig.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])

                        
                    else:
                        for polytope in pambig:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])  
                                
                    return ambiguous_regions, negative_regions
        
                    
            vertices1=np.array(vertices1)
            vertices2=np.array(vertices2)
            
            
            
            
            if (len(vertices1)!=0) and (len(vertices2)!=0): 
                # Compute the two polytopes p1 and p2 using the H-representations
                p1 = pc.qhull(vertices1)
                p2 = pc.qhull(vertices2)          
                # Compute H-representations of  p1 and p2
                A1=p1.A
                b1=p1.b
                A2=p2.A
                b2=p2.b               
                # Compute the polytope p3 the union of p1 and p2 (it presents the negative region)
                if ((p1.volume==0) and (p2.volume==0)):
                    p1,p2=self.Part_polype(polype)
                    pambig=p1.union(p2)
                    
                    if len(pambig)==0:
                        if pambig.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])

                        
                    else:
                        for polytope in pambig:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])  
                                
                    return ambiguous_regions, negative_regions
                    
                    
                elif ((p1.volume!=0) and (p2.volume==0)):
                    # obatin the underapprox rect 1
                    boxunrec1=self.RecinPolytope(A1,b1) 
                    boxunrec1=np.append(boxunrec1[0], boxunrec1[1], axis=1) 
                    p1=pc.box2poly(boxunrec1)
                    negative_regions.append([{'A':p1.A,'b':p1.b}])
                    # Compute the polytope p4 which presents the ambiguous region (pregion-p1)
                    p4=polype.diff(p1) 
                    
                elif ((p1.volume==0) and (p2.volume!=0)):
                    # obatin the underapprox rect 2
                    boxunrec2=self.RecinPolytope(A2,b2)
                    boxunrec2=np.append(boxunrec2[0], boxunrec2[1], axis=1) 
                    p2=pc.box2poly(boxunrec2)
                    negative_regions.append([{'A':p2.A,'b':p2.b}])
                    # Compute the polytope p4 which presents the ambiguous region (pregion-p2)
                    p4=polype.diff(p2)
                    
                elif ((p1.volume!=0) and (p2.volume!=0)):
                    # obatin the underapprox rect 1
                    boxunrec1=self.RecinPolytope(A1,b1)
                    boxunrec1=np.append(boxunrec1[0], boxunrec1[1], axis=1)  
                    p1=pc.box2poly(boxunrec1)
                    negative_regions.append([{'A':p1.A,'b':p1.b}])
                    
                    # obatin the underapprox rect 2
                    boxunrec2=self.RecinPolytope(A2,b2)
                    boxunrec2=np.append(boxunrec2[0], boxunrec2[1], axis=1)  
                    p2=pc.box2poly(boxunrec2)
                    negative_regions.append([{'A':p2.A,'b':p2.b}])
                    p3=p2.union(p1)             
                    # Compute the polytope p4 which presents the ambiguous region (pregion-p3)
                    p4=polype.diff(p3) 
                    
                    
 
                # Compute the H-repren of each polytope in p4
                if len(p4)==0:
                    if p4.volume==0:
                        ambiguous_regions=ambiguous_regions+[]
                    else:
                        ambiguous_regions.append([{'A':p4.A,'b':p4.b}]) 
                else:
                    for polytope in p4:
                        if polytope.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}]) 
                
                return ambiguous_regions, negative_regions
            
            elif (len(vertices1)!=0) and (len(vertices2)==0):           
                # Compute the polytope p1 
                p1= pc.qhull(vertices1) 
                # Compute H-representations of  p1 
                A1=p1.A
                b1=p1.b   
                
                if p1.volume==0:
                    p1,p2=self.Part_polype(polype)
                    pambig=p1.union(p2)
                    
                    if len(pambig)==0:
                        if pambig.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])

                        
                    else:
                        for polytope in pambig:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])  
                                
                    return ambiguous_regions, negative_regions
                    
                    
                    
                else:
                    # obatin the underapprox rect 1
                    boxunrec1=self.RecinPolytope(A1,b1)
                    boxunrec1=np.append(boxunrec1[0], boxunrec1[1], axis=1) 
                    p1=pc.box2poly(boxunrec1)
                    negative_regions.append([{'A':p1.A,'b':p1.b}])                    
                    # Compute the polytope p4 which presents the ambiguous region (pregion-p1)
                    p4=polype.diff(p1)     
                    # Compute the H-repren of each polytope in p4
                    if len(p4)==0:
                        if p4.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':p4.A,'b':p4.b}]) 
                    else:
                        for polytope in p4:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}]) 
                                
                    return ambiguous_regions, negative_regions

                
            elif (len(vertices1)==0) and (len(vertices2)!=0):     
                # Compute the polytope p2 
                p2= pc.qhull(vertices2) 
                # Compute H-representations of  p2 
                A2=p2.A
                b2=p2.b   
                
                if p2.volume==0:
                    p1,p2=self.Part_polype(polype)
                    pambig=p1.union(p2)
                    
                    if len(pambig)==0:
                        if pambig.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])

                        
                    else:
                        for polytope in pambig:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])  
                                
                    return ambiguous_regions, negative_regions
                    
                    
                    
                else:
                    # obatin the underapprox rect 2
                    boxunrec2=self.RecinPolytope(A2,b2)
                    boxunrec2=np.append(boxunrec2[0], boxunrec2[1], axis=1) 
                    p2=pc.box2poly(boxunrec2)
                    negative_regions.append([{'A':p2.A,'b':p2.b}])                    
                    # Compute the polytope p4 which presents the ambiguous region (pregion-p2)
                    p4=polype.diff(p2)     
                    # Compute the H-repren of each polytope in p4
                    if len(p4)==0:
                        if p4.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':p4.A,'b':p4.b}]) 
                    else:
                        for polytope in p4:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}]) 
                                
                    return ambiguous_regions, negative_regions
 
 


        else: # There is only one sheet  
            # print('1sheet')
            # Compute 2n vertices (2n faces) of the polytope that under-approximate the negative region if it exists         
            # Construct 2n template vectors 
            Tem1=np.identity(self.num_vars)
            Tem2=-np.identity(self.num_vars)
            Tem=np.insert(Tem1,self.num_vars,Tem2,axis=0)
        
            # Compute the vertices   
            vertices=[]
            for i in range((self.num_vars)+1):
                # print('kkkkkkkkkkkkkk3'+str(i))

                v=self.Ver_tang_one_sheet(poly,pregion,mid_point,Hess,Gradi,Rem2,Tem[i,:],'N')
            
            
                # if the vertice exist==> Neg Reg exist==> Save it
                if len(v)!=0:
                    vertices.append(v)
                else: # if the vertice  does not exist==> Neg Reg does not exist==> Partition Ambiguous Region
                    # create a polytope that is inside pregion formed by a center and n vertices
                    p1,p2=self.Part_polype(polype)
                    pambig=p1.union(p2)
                    # Compute Aambig and bambig of each region in the partition region
                    if len(pambig)==0:
                        if pambig.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])

                        
                    else:
                        for polytope in pambig:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])                   
                    return ambiguous_regions, negative_regions
             
                
 
            vertices=np.array(vertices)   
            p1=pc.qhull(vertices) 
            A1=p1.A
            b1=p1.b      
            if p1.volume==0:
                
                    p1,p2=self.Part_polype(polype)
                    pambig=p1.union(p2)
                    # Compute Aambig and bambig of each region in the partition region
                    if len(pambig)==0:
                        if pambig.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])

                        
                    else:
                        for polytope in pambig:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])                   
                    return ambiguous_regions, negative_regions
                
            else:
                    # obatin the underapprox rect 1
                    boxunrec1=self.RecinPolytope(A1,b1)
                    boxunrec1=np.append(boxunrec1[0], boxunrec1[1], axis=1) 
                    p1=pc.box2poly(boxunrec1)
                    negative_regions.append([{'A':p1.A,'b':p1.b}]) 
                        
                    p4=polype.diff(p1) 
                    if len(p4)==0:
                        if p4.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':p4.A,'b':p4.b}]) 
                    else:
                        for polytope in p4:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])             
                    return ambiguous_regions, negative_regions
        
        
        
        
        
        
        
        
    # ========================================================
    # Partition the region (Pos+Ambig) based on the polynomial sign
    # ========================================================  
    def Ppartition_inequality_regions(self,inequality_index, pregion): 
        num_samples=3
        poly=self.poly_inequality_coeffs[inequality_index]
        
        # Initialization of the pos, ambig, regions
        positive_regions        = []
        ambiguous_regions       = [] 
        # Transform pregion into polytope format
        Areg=pregion[0]['A']
        breg=pregion[0]['b']
        polype=pc.Polytope(Areg, breg) 
        
        
        # Compute the middle point in the polytope region
        rb,mid_point=pc.cheby_ball(polype)
     
        # Compute the hessian Matrix of Poly at mid_point
        Hess=self.Hessian(poly,mid_point)
    
        # Compute the gradient vector of Poly at mid_point
        Gradi=self.Gradient(poly,mid_point)
        
        # Compute the remainder for the 2nd order Taylor overapproximation
        Rem2=self.remainder2cst(poly,pregion,mid_point,Gradi,Hess)
        
        if self.num_pos_eig(Hess)==1: # There is two sheets
            # Compute the hyperplane (As,bs) that will separate the two sheets
              
            #  Compute the center of the hyperbola
            coefcen=Hess.dot(mid_point)-Gradi
            cen=np.linalg.solve(Hess,coefcen)
            # Compute the principal axis As
            eigval,eigvec=np.linalg.eig(Hess)
            # Compute the index of the eigenvalue > 0
            indexaux=np.nonzero(eigval> 0)
            index=indexaux[0][0]
            # Compute As and bs
            As=eigvec[index,:]
            bs=As.dot(cen)
             
            # Compute 2n vertices (2n faces) of the two polytope that under-approximate the negative regions if they exist      
            # Construct 2n template vectors 
            Tem1=np.identity(self.num_vars)
            Tem2=-np.identity(self.num_vars)
            Tem=np.insert(Tem1,self.num_vars,Tem2,axis=0)
        
            # Compute the vertices of the two polytopes  
            vertices1=[]
            vertices2=[]
            for i in range(2*(self.num_vars)):

                v1=self.Ver_tang_two_sheet(poly,pregion,mid_point,Hess,Gradi,Rem2,Tem[i,:],As,bs,'P')
                v2=self.Ver_tang_two_sheet(poly,pregion,mid_point,Hess,Gradi,Rem2,Tem[i,:],-As,-bs,'P')
                v1=list(v1)
                v2=list(v2)
            
            
                # if the vertices exist==> Neg Reg exist==> Save it
                # print(v1,v2)
                if (len(v1)!=0) and (len(v2)!=0):
                    vertices1.append(v1)
                    vertices2.append(v2)
                elif (len(v1)!=0):  
                    vertices1.append(v1)
                elif (len(v2)!=0):  
                    vertices2.append(v2)    
                else: # if the vertices  does not exist==> Pos Reg does not exist==> Partition Ambiguous Region
                    # create a polytope that is inside pregion formed by a center and n vertices
                    p1,p2=self.Part_polype(polype)
                    pambig=p1.union(p2)
                    
                    if len(pambig)==0:
                        if pambig.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])

                        
                    else:
                        for polytope in pambig:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])  
                                
                    return ambiguous_regions, positive_regions
        
                    
            vertices1=np.array(vertices1)
            vertices2=np.array(vertices2)
            
            
            
            if (len(vertices1)!=0) and (len(vertices2)!=0): 
                # Compute the two polytopes p1 and p2 using the H-representations
                p1 = pc.qhull(vertices1)
                p2 = pc.qhull(vertices2)          
                # Compute H-representations of  p1 and p2
                A1=p1.A
                b1=p1.b
                A2=p2.A
                b2=p2.b               
                # Compute the polytope p3 the union of p1 and p2 (it presents the negative region)
                if ((p1.volume==0) and (p2.volume==0)):
                    p1,p2=self.Part_polype(polype)
                    pambig=p1.union(p2)
                    
                    if len(pambig)==0:
                        if pambig.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])

                        
                    else:
                        for polytope in pambig:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])  
                                
                    return ambiguous_regions, positive_regions
                    
                    
                elif ((p1.volume!=0) and (p2.volume==0)):
                    # obatin the underapprox rect 1
                    boxunrec1=self.RecinPolytope(A1,b1)
                    boxunrec1=np.append(boxunrec1[0], boxunrec1[1], axis=1)  
                    p1=pc.box2poly(boxunrec1)
                    positive_regions.append([{'A':p1.A,'b':p1.b}])
                    # Compute the polytope p4 which presents the ambiguous region (pregion-p1)
                    p4=polype.diff(p1) 
                    
                elif ((p1.volume==0) and (p2.volume!=0)):
                    # obatin the underapprox rect 2
                    boxunrec2=self.RecinPolytope(A2,b2)
                    boxunrec2=np.append(boxunrec2[0], boxunrec2[1], axis=1) 
                    p2=pc.box2poly(boxunrec2)
                    positive_regions.append([{'A':p2.A,'b':p2.b}])
                    # Compute the polytope p4 which presents the ambiguous region (pregion-p2)
                    p4=polype.diff(p2)
                    
                elif ((p1.volume!=0) and (p2.volume!=0)):
                    # obatin the underapprox rect 1
                    boxunrec1=self.RecinPolytope(A1,b1)
                    boxunrec1=np.append(boxunrec1[0], boxunrec1[1], axis=1) 
                    p1=pc.box2poly(boxunrec1)
                    positive_regions.append([{'A':p1.A,'b':p1.b}])
                    
                    # obatin the underapprox rect 2
                    boxunrec2=self.RecinPolytope(A2,b2)
                    boxunrec1=np.append(boxunrec2[0], boxunrec2[1], axis=1) 
                    p2=pc.box2poly(boxunrec2)
                    positive_regions.append([{'A':p2.A,'b':p2.b}])
                    p3=p2.union(p1)             
                    # Compute the polytope p4 which presents the ambiguous region (pregion-p3)
                    p4=polype.diff(p3) 
                    
                    
 
                # Compute the H-repren of each polytope in p4
                if len(p4)==0:
                    if p4.volume==0:
                        ambiguous_regions=ambiguous_regions+[]
                    else:
                        ambiguous_regions.append([{'A':p4.A,'b':p4.b}]) 
                else:
                    for polytope in p4:
                        if polytope.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}]) 
                
                return ambiguous_regions, positive_regions
            
            elif (len(vertices1)!=0) and (len(vertices2)==0):           
                # Compute the polytope p1 
                p1= pc.qhull(vertices1) 
                # Compute H-representations of  p1 
                A1=p1.A
                b1=p1.b   
                
                if p1.volume==0:
                    p1,p2=self.Part_polype(polype)
                    pambig=p1.union(p2)
                    
                    if len(pambig)==0:
                        if pambig.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])

                        
                    else:
                        for polytope in pambig:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])  
                                
                    return ambiguous_regions, positive_regions
                    
                    
                    
                else:
                    # obatin the underapprox rect 1
                    boxunrec1=self.RecinPolytope(A1,b1)
                    boxunrec1=np.append(boxunrec1[0], boxunrec1[1], axis=1) 
                    p1=pc.box2poly(boxunrec1)
                    positive_regions.append([{'A':p1.A,'b':p1.b}])                    
                    # Compute the polytope p4 which presents the ambiguous region (pregion-p1)
                    p4=polype.diff(p1)     
                    # Compute the H-repren of each polytope in p4
                    if len(p4)==0:
                        if p4.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':p4.A,'b':p4.b}]) 
                    else:
                        for polytope in p4:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}]) 
                                
                    return ambiguous_regions, positive_regions

                
            elif (len(vertices1)==0) and (len(vertices2)!=0):     
                # Compute the polytope p2 
                p2= pc.qhull(vertices2) 
                # Compute H-representations of  p2 
                A2=p2.A
                b2=p2.b   
                
                if p2.volume==0:
                    p1,p2=self.Part_polype(polype)
                    pambig=p1.union(p2)
                    
                    if len(pambig)==0:
                        if pambig.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])

                        
                    else:
                        for polytope in pambig:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])  
                                
                    return ambiguous_regions, positive_regions
                    
                    
                    
                else:
                    # obatin the underapprox rect 2
                    boxunrec2=self.RecinPolytope(A2,b2)
                    boxunrec2=np.append(boxunrec2[0], boxunrec2[1], axis=1) 
                    p2=pc.box2poly(boxunrec2)
                    positive_regions.append([{'A':p2.A,'b':p2.b}])                    
                    # Compute the polytope p4 which presents the ambiguous region (pregion-p2)
                    p4=polype.diff(p2)     
                    # Compute the H-repren of each polytope in p4
                    if len(p4)==0:
                        if p4.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':p4.A,'b':p4.b}]) 
                    else:
                        for polytope in p4:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}]) 
                                
                    return ambiguous_regions, positive_regions
 
 


        else: # There is only one sheet  
            # print('kkkkkkkkkkkkkk2')
            # Compute 2n vertices (2n faces) of the polytope that under-approximate the negative region if it exists         
            # Construct 2n template vectors 
            Tem1=np.identity(self.num_vars)
            Tem2=-np.identity(self.num_vars)
            Tem=np.insert(Tem1,self.num_vars,Tem2,axis=0)
        
            # Compute the vertices   
            vertices=[]
            for i in range((self.num_vars)+1):
                # print('kkkkkkkkkkkkkk3'+str(i))

                v=self.Ver_tang_one_sheet(poly,pregion,mid_point,Hess,Gradi,Rem2,Tem[i,:],'N')
            
            
                # if the vertice exist==> Neg Reg exist==> Save it
                if len(v)!=0:
                    vertices.append(v)
                else: # if the vertice  does not exist==> Neg Reg does not exist==> Partition Ambiguous Region
                    # create a polytope that is inside pregion formed by a center and n vertices
                    p1,p2=self.Part_polype(polype)
                    pambig=p1.union(p2)
                    # Compute Aambig and bambig of each region in the partition region
                    if len(pambig)==0:
                        if pambig.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])

                        
                    else:
                        for polytope in pambig:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])                   
                    return ambiguous_regions, positive_regions
             
                
 
            vertices=np.array(vertices)     
            p1=pc.qhull(vertices) 
            A1=p1.A
            b1=p1.b      
            if p1.volume==0:
                
                    p1,p2=self.Part_polype(polype)
                    pambig=p1.union(p2)
                    # Compute Aambig and bambig of each region in the partition region
                    if len(pambig)==0:
                        if pambig.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])

                        
                    else:
                        for polytope in pambig:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])                   
                    return ambiguous_regions, positive_regions
                
            else:
                    # obatin the underapprox rect 1
                    boxunrec1=self.RecinPolytope(A1,b1)
                    boxunrec1=np.append(boxunrec1[0], boxunrec1[1], axis=1) 
                    p1=pc.box2poly(boxunrec1)
                    positive_regions.append([{'A':p1.A,'b':p1.b}]) 
                         
                    p4=polype.diff(p1) 
                    if len(p4)==0:
                        if p4.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':p4.A,'b':p4.b}]) 
                    else:
                        for polytope in p4:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])             
                    return ambiguous_regions, positive_regions
            
            
        
    
    

     
    
    # ========================================================
    #   Use Z3 to Solve many multivariable Polynomial Constraints
    # ========================================================
     
    def solveZ3_many_multivars(self,polys, box,q):
        solver = z3.Solver()
        x = z3.Reals(self.strVari(self.num_vars))
        
        
        for i in range(self.num_vars):
            solver.add(x[i]>=(box[0][i][0]))
            solver.add(x[i]<=box[1][i][0])
        

        polycs=[]

        for poly in polys:
            poly_constraint = 0
            for monomial_counter in range(0,len(poly)):
                coeff = poly[monomial_counter]['coeff']
                vars  = poly[monomial_counter]['vars']
                product = coeff

                zc=0
                for var_counter in range(len(vars)):
                    power = vars[var_counter]['power']
                    if power==0:
                        zc=zc+1
                    else:    
                        var   = x[var_counter]  
                        product = product * (var**power)
                if zc==len(vars):
                    poly_constraint = poly_constraint + coeff
                else:
                    poly_constraint = poly_constraint + product
            solver.add(poly_constraint <= 0)
            polycs.append(poly_constraint)
        
        if solver.check() == sat:
            model = solver.model()
            q.put(model[x[3]].as_fraction())
            return 'SAT'
        
        return 'UNSAT' 
    
    
    # ========================================================
    #   Use Z3 to Solve many multivariable Polynomial Constraints
    # ========================================================
     
    def solveZ3(self,polys, box):
        solver = z3.Solver()
        x = z3.Reals(self.strVari(self.num_vars))
        
        
        for i in range(self.num_vars):
            solver.add(x[i]>=(box[0][i][0]))
            solver.add(x[i]<=box[1][i][0])
        

        polycs=[]

        for poly in polys:
            poly_constraint = 0
            for monomial_counter in range(0,len(poly)):
                coeff = poly[monomial_counter]['coeff']
                vars  = poly[monomial_counter]['vars']
                product = coeff

                zc=0
                for var_counter in range(len(vars)):
                    power = vars[var_counter]['power']
                    if power==0:
                        zc=zc+1
                    else:    
                        var   = x[var_counter]  
                        product = product * (var**power)
                if zc==len(vars):
                    poly_constraint = poly_constraint + coeff
                else:
                    poly_constraint = poly_constraint + product
            solver.add(poly_constraint <= 0)
            polycs.append(poly_constraint)
        
        if solver.check() == sat:
            model = solver.model()
            # q.put('SAT')
            # print(type(model[x[2]].as_fraction()))
            return 'SAT', model[x[3]].as_fraction()
        
        return 'UNSAT', 0 
    
    
    

    
    
    # ========================================================
    #   Use Yices to Solve many multivariable 
    #          Polynomial Constraints
    # ========================================================
     
    def Yicesmany_multivars(self,polys,box,q):
    
        cfg = Config()
        cfg.default_config_for_logic('QF_NRA')
        ctx = Context(cfg)
        real_t = Types.real_type()
        
        

        # print('islem')

        
        x0 = Terms.new_uninterpreted_term(real_t, 'x0')
        x1 = Terms.new_uninterpreted_term(real_t, 'x1')
        x2 = Terms.new_uninterpreted_term(real_t, 'x2')
        x3 = Terms.new_uninterpreted_term(real_t, 'x3')
        x4 = Terms.new_uninterpreted_term(real_t, 'x4')
        x5 = Terms.new_uninterpreted_term(real_t, 'x5')
        x6 = Terms.new_uninterpreted_term(real_t, 'x6')
        x7 = Terms.new_uninterpreted_term(real_t, 'x7')
        x8 = Terms.new_uninterpreted_term(real_t, 'x8')
        x9 = Terms.new_uninterpreted_term(real_t, 'x9')
        x10 = Terms.new_uninterpreted_term(real_t, 'x10')
        x11 = Terms.new_uninterpreted_term(real_t, 'x11')
        x12 = Terms.new_uninterpreted_term(real_t, 'x12')
        x13 = Terms.new_uninterpreted_term(real_t, 'x13')
        x14 = Terms.new_uninterpreted_term(real_t, 'x14')
        x15 = Terms.new_uninterpreted_term(real_t, 'x15')
        x16 = Terms.new_uninterpreted_term(real_t, 'x16')
        x17 = Terms.new_uninterpreted_term(real_t, 'x17')
        x18 = Terms.new_uninterpreted_term(real_t, 'x18')
        x19 = Terms.new_uninterpreted_term(real_t, 'x19')
        x20 = Terms.new_uninterpreted_term(real_t, 'x20')
        x21 = Terms.new_uninterpreted_term(real_t, 'x21')
        x22 = Terms.new_uninterpreted_term(real_t, 'x22')
        x23 = Terms.new_uninterpreted_term(real_t, 'x23')
        x24 = Terms.new_uninterpreted_term(real_t, 'x24')
        x25 = Terms.new_uninterpreted_term(real_t, 'x25')
        x26 = Terms.new_uninterpreted_term(real_t, 'x26')
        x27 = Terms.new_uninterpreted_term(real_t, 'x27')
        x28 = Terms.new_uninterpreted_term(real_t, 'x28')
        x29 = Terms.new_uninterpreted_term(real_t, 'x29')
        x30 = Terms.new_uninterpreted_term(real_t, 'x30')
        x31 = Terms.new_uninterpreted_term(real_t, 'x31')
        x32 = Terms.new_uninterpreted_term(real_t, 'x32')
        x33 = Terms.new_uninterpreted_term(real_t, 'x33')
        x34 = Terms.new_uninterpreted_term(real_t, 'x34')
        x35 = Terms.new_uninterpreted_term(real_t, 'x35')
        x36 = Terms.new_uninterpreted_term(real_t, 'x36')
        x37 = Terms.new_uninterpreted_term(real_t, 'x37')
        x38 = Terms.new_uninterpreted_term(real_t, 'x38')
        x39 = Terms.new_uninterpreted_term(real_t, 'x39')
        x40 = Terms.new_uninterpreted_term(real_t, 'x40')
        x41 = Terms.new_uninterpreted_term(real_t, 'x41')
        x42 = Terms.new_uninterpreted_term(real_t, 'x42')
        x43 = Terms.new_uninterpreted_term(real_t, 'x43')
        x44 = Terms.new_uninterpreted_term(real_t, 'x44')
        x45 = Terms.new_uninterpreted_term(real_t, 'x45')
        x46 = Terms.new_uninterpreted_term(real_t, 'x46')
        x47 = Terms.new_uninterpreted_term(real_t, 'x47')
        x48 = Terms.new_uninterpreted_term(real_t, 'x48')
        x49 = Terms.new_uninterpreted_term(real_t, 'x49')
        x50 = Terms.new_uninterpreted_term(real_t, 'x50')
        x51 = Terms.new_uninterpreted_term(real_t, 'x51')
        x52 = Terms.new_uninterpreted_term(real_t, 'x52')
        x53 = Terms.new_uninterpreted_term(real_t, 'x53')
        x54 = Terms.new_uninterpreted_term(real_t, 'x54')
        x55 = Terms.new_uninterpreted_term(real_t, 'x55')
        x56 = Terms.new_uninterpreted_term(real_t, 'x56')
        x57 = Terms.new_uninterpreted_term(real_t, 'x57')
        x58 = Terms.new_uninterpreted_term(real_t, 'x58')
        x59 = Terms.new_uninterpreted_term(real_t, 'x59')
        x60 = Terms.new_uninterpreted_term(real_t, 'x60')

        fmlaliststr=[]
        fmlalist=[]
        for poly in polys:
            res=self.fmlapoly(poly)
            fmlaliststr.append(res)
            
        fmlaliststr.append(self.fmlabounds2(box)) 
        for i in range(len(fmlaliststr)):
            fmla=Terms.parse_term(fmlaliststr[i])
            fmlalist.append(fmla)
            

        

        ctx.assert_formulas(fmlalist)
        status = ctx.check_context()
        if status == Status.SAT:
            model = Model.from_context(ctx, 1)
            model_string = model.to_string(80, 100, 0)
            # sol=np.array([model.get_value(x0),model.get_value(x1),model.get_value(x2)])
            # q.put(sol)
            u0=model.get_value(x3)
            q.put(u0)
            return 'SAT', u0
        
        else:
            return 'UNSAT'
        
        
        
        
        
        
    # ========================================================
    #   Use Yices to Solve many multivariable 
    #          Polynomial Constraints
    # ========================================================
     
    def solveYices(self,polys,box):
    
        cfg = Config()
        cfg.default_config_for_logic('QF_NRA')
        ctx = Context(cfg)
        real_t = Types.real_type()
        
        
        
        

        
        x0 = Terms.new_uninterpreted_term(real_t, 'x0')
        x1 = Terms.new_uninterpreted_term(real_t, 'x1')
        x2 = Terms.new_uninterpreted_term(real_t, 'x2')
        x3 = Terms.new_uninterpreted_term(real_t, 'x3')
        x4 = Terms.new_uninterpreted_term(real_t, 'x4')
        x5 = Terms.new_uninterpreted_term(real_t, 'x5')
        x6 = Terms.new_uninterpreted_term(real_t, 'x6')
        x7 = Terms.new_uninterpreted_term(real_t, 'x7')
        x8 = Terms.new_uninterpreted_term(real_t, 'x8')
        x9 = Terms.new_uninterpreted_term(real_t, 'x9')
        x10 = Terms.new_uninterpreted_term(real_t, 'x10')
        x11 = Terms.new_uninterpreted_term(real_t, 'x11')
        x12 = Terms.new_uninterpreted_term(real_t, 'x12')
        x13 = Terms.new_uninterpreted_term(real_t, 'x13')
        x14 = Terms.new_uninterpreted_term(real_t, 'x14')
        x15 = Terms.new_uninterpreted_term(real_t, 'x15')
        x16 = Terms.new_uninterpreted_term(real_t, 'x16')
        x17 = Terms.new_uninterpreted_term(real_t, 'x17')
        x18 = Terms.new_uninterpreted_term(real_t, 'x18')
        x19 = Terms.new_uninterpreted_term(real_t, 'x19')
        x20 = Terms.new_uninterpreted_term(real_t, 'x20')
        x21 = Terms.new_uninterpreted_term(real_t, 'x21')
        x22 = Terms.new_uninterpreted_term(real_t, 'x22')
        x23 = Terms.new_uninterpreted_term(real_t, 'x23')
        x24 = Terms.new_uninterpreted_term(real_t, 'x24')
        x25 = Terms.new_uninterpreted_term(real_t, 'x25')
        x26 = Terms.new_uninterpreted_term(real_t, 'x26')
        x27 = Terms.new_uninterpreted_term(real_t, 'x27')
        x28 = Terms.new_uninterpreted_term(real_t, 'x28')
        x29 = Terms.new_uninterpreted_term(real_t, 'x29')
        x30 = Terms.new_uninterpreted_term(real_t, 'x30')
        x31 = Terms.new_uninterpreted_term(real_t, 'x31')
        x32 = Terms.new_uninterpreted_term(real_t, 'x32')
        x33 = Terms.new_uninterpreted_term(real_t, 'x33')
        x34 = Terms.new_uninterpreted_term(real_t, 'x34')
        x35 = Terms.new_uninterpreted_term(real_t, 'x35')
        x36 = Terms.new_uninterpreted_term(real_t, 'x36')
        x37 = Terms.new_uninterpreted_term(real_t, 'x37')
        x38 = Terms.new_uninterpreted_term(real_t, 'x38')
        x39 = Terms.new_uninterpreted_term(real_t, 'x39')
        x40 = Terms.new_uninterpreted_term(real_t, 'x40')
        x41 = Terms.new_uninterpreted_term(real_t, 'x41')
        x42 = Terms.new_uninterpreted_term(real_t, 'x42')
        x43 = Terms.new_uninterpreted_term(real_t, 'x43')
        x44 = Terms.new_uninterpreted_term(real_t, 'x44')
        x45 = Terms.new_uninterpreted_term(real_t, 'x45')
        x46 = Terms.new_uninterpreted_term(real_t, 'x46')
        x47 = Terms.new_uninterpreted_term(real_t, 'x47')
        x48 = Terms.new_uninterpreted_term(real_t, 'x48')
        x49 = Terms.new_uninterpreted_term(real_t, 'x49')
        x50 = Terms.new_uninterpreted_term(real_t, 'x50')
        x51 = Terms.new_uninterpreted_term(real_t, 'x51')
        x52 = Terms.new_uninterpreted_term(real_t, 'x52')
        x53 = Terms.new_uninterpreted_term(real_t, 'x53')
        x54 = Terms.new_uninterpreted_term(real_t, 'x54')
        x55 = Terms.new_uninterpreted_term(real_t, 'x55')
        x56 = Terms.new_uninterpreted_term(real_t, 'x56')
        x57 = Terms.new_uninterpreted_term(real_t, 'x57')
        x58 = Terms.new_uninterpreted_term(real_t, 'x58')
        x59 = Terms.new_uninterpreted_term(real_t, 'x59')
        x60 = Terms.new_uninterpreted_term(real_t, 'x60')

        fmlaliststr=[]
        fmlalist=[]
        for poly in polys:
            res=self.fmlapoly(poly)
            fmlaliststr.append(res)
            
        fmlaliststr.append(self.fmlabounds2(box)) 
        for i in range(len(fmlaliststr)):
            fmla=Terms.parse_term(fmlaliststr[i])
            fmlalist.append(fmla)
            

        

        ctx.assert_formulas(fmlalist)
        status = ctx.check_context()
        if status == Status.SAT:
            model = Model.from_context(ctx, 1)
            model_string = model.to_string(80, 100, 0)
            # sol=np.array([model.get_value(x0),model.get_value(x1),model.get_value(x2)])
            # q.put(sol)
            u0=model.get_value(x3)
            return 'SAT', u0
        
        else:
            return 'UNSAT'





# ========================================================
#   Function to output the n^th dimenstion hypercube
#      with edge limited between xmin and xmax
# ========================================================
def hypercube(n, xmin, xmax):
    box=[]
    for i in range(n):
        box.append([xmin,xmax])
    
    return box         
        
if __name__ == "__main__":

    num_vars = 2
    
    x_min = -1.0
    x_max = 1.0
    
    
    box=np.array(hypercube(num_vars, x_min,x_max))
    polype=pc.box2poly(box)
    boxx=pc.bounding_box(polype)
    pregion=[[{'A':polype.A,'b':polype.b}]]
    orders = [[2, 2]]
    solver = PolyInequalitySolver(num_vars, boxx, pregion, orders)
    
    # poly = 4 x^2 + 3 y^2 + 2
    poly = [
    {'coeff':4,      'vars':[{'power':2},{'power':0}]},
    {'coeff':3,    'vars':[{'power':0},{'power':2}]},
    {'coeff':2,    'vars':[{'power':0},{'power':0}]}
    ]
    
    

    
    solver.addPolyInequalityConstraint(poly)
    
    start_time = time.time()
    res=solver.solve()   

    print(res)    
   


        

        
        
        
        

    
        
        
        
        
