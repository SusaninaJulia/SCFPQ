import sys, os, copy
import numpy as np


import cupy as cp
from cupyx.scipy import sparse

from newton_krylov import nonlin_solve


class Grammar:
    
    def __init__(self, file_path, grammar_type = 'bool'):
        
        self.file_path = file_path
        
        self.rules = dict()
        self.flag = False if grammar_type == 'bool' else True  # False if grammar is Bool, True -- if Probabilistic 
        
        nt = dict() # nonterminals
        t = dict() # terminals
        
        ntc = 0
        tc = 0
                
        with open(self.file_path, 'r') as f:
            for line in f:

                (l, r) = line.split(' -> ') 
                
                if l in nt.keys():
                    l = nt[l]
                else:
                    nt[l] = ntc + tc
                    l = ntc + tc
                    ntc = ntc + 1
                
                r = r.split()
                
                if not self.flag:
                    r = [*r, 1]
                    
                r[-1] = float(r[-1])
                
                if len(r) == 2:
                    r[0] = r[0][4:]
                    if r[0] in t.keys():
                        r[0] = t[r[0]]
                    else:
                        t[r[0]] = ntc + tc
                        r[0] = ntc + tc
                        tc = tc + 1
                    
                if len(r) == 3:

                    for i in range(2):
                        if 'VAR:' in r[i]:
                            r[i] = r[i][4:]
                            if r[i] in t.keys():
                                r[i] = t[r[i]]
                            else:
                                t[r[i]] = ntc + tc
                                r[i] = ntc + tc
                                tc = tc + 1

                        elif r[i] in nt.keys(): 
                            r[i] = nt[r[i]]
                        else:
                            nt[r[i]] = ntc + tc
                            r[i] = ntc + tc
                            ntc = ntc + 1
                    
                if l in self.rules.keys():
                    self.rules[l].append(r) 
                else:
                    self.rules[l] = [r]
                        
        self.ntc = ntc # nonterminals count
        self.nonterminals = nt #{i : nt.pop() for i in range(self.ntc)}
        
        self.tc = tc # terminals count
        self.terminals = t #{i : t.pop() for i in range(self.tc)}
        
        print(self.nonterminals, self.ntc)
        print(self.terminals, self.tc)
        print(self.rules)
        
        

    
class Graph:
    
    def __init__(self, file_path):
        
        self.file_path = file_path
        self.en = 0 # number of edges
        self.flag = False # False if graph is Bool, True -- if Probabilistic 
        self.matrices = dict()

        with open(self.file_path, 'r') as f:
            for line in f:
                edge = line.split()
                if len(edge) == 4:
                    self.flag = True
                self.en = max([int(edge[0]), int(edge[2]), self.en])
        self.en = self.en + 1
                
        
        
    def fill(self, grammar):
        
        alp = grammar.tc + grammar.ntc
        
        self.matrices = dict()
        
        row_ind = {i : [] for i in range(alp)}
        col_ind = {i : [] for i in range(alp)}
        data = {i : [] for i in range(alp)}
                
        with open(self.file_path, 'r') as f:
            for line in f:
                edge = line.split()
                
                if edge[1] in grammar.terminals.keys():
                    ind = grammar.terminals[edge[1]]
                else:
                    continue
                    
                row_ind[ind].append(int(edge[0]))
                col_ind[ind].append(int(edge[2]))
                    
                if self.flag:
                    data[ind].append(float(edge[3]))
                else:
                    data[ind].append(1.0)
        
        for ind in range(0, alp):  
            
            d = cp.array(data[ind])
            r = cp.array(row_ind[ind])
            c = cp.array(col_ind[ind])

            sp = sparse.csr_matrix((d, (r, c)), shape=(self.en, self.en))
            
            self.matrices[ind] = sp
        
        self.matrices[alp] = sparse.csr_matrix((self.en, self.en))#sparse.csr_matrix((cp.array([]),(cp.array([]),cp.array([]))),shape=(self.en, self.en))

    def make_numpy(self):
        for i in self.matrices.keys():
            self.matrices[i] = cp.asnumpy(self.matrices[i])

class Equation:
    
    def __init__(self, grammar, graph):
        
        self.graph = graph
        self.en = graph.en
        
        self.grammar = grammar
        self.rules = grammar.rules
        self.S = grammar.nonterminals['S']
        self.alp = grammar.tc + grammar.ntc
        
        self.matrices = graph.matrices

    def ni_equation(self, prev_step, next_step):

        tmp = sparse.csr_matrix((self.en, self.en))
        for l in self.rules.keys():
            r = self.rules[l]
            for r_i in r:
                if len(r_i) == 2:
                    tmp = tmp + r_i[1] * prev_step[r_i[0]]
                else:
                    tmp = tmp + r_i[2] * prev_step[r_i[0]].dot(prev_step[r_i[1]])
            next_step[l] = tmp
            tmp = sparse.csr_matrix((self.en, self.en))
        return next_step


    def naive_iteration(self, initial_guess=None, equation=None, tol=10e-15, info = True):

        prev_step = dict()
        if initial_guess is None:
            for ind in self.grammar.nonterminals.values():
                prev_step[ind] = sparse.csr_matrix((self.en, self.en))
        else:
            for i, ind in enumerate(self.grammar.nonterminals.values()):
                prev_step[ind] = initial_guess[ind]
        for ind in self.grammar.terminals.values():
            prev_step[ind] = self.matrices[ind]

        next_step = dict()
        for ind in self.grammar.nonterminals.values():
            next_step[ind] = prev_step[ind].copy()

        if equation is None:
            equation = self.ni_equation

        start = self.grammar.nonterminals['S']
        i = 1

        def norm():
            na = np.empty(self.grammar.ntc)
            for i, ind in enumerate(self.grammar.nonterminals.values()):
                na[i] = (sparse.linalg.norm(next_step[ind] - prev_step[ind]))
            return np.linalg.norm(na)

        if info:
            print('step: ', i, '#elements: ', prev_step[start].count_nonzero())

        while True:

            next_step = equation(prev_step, next_step)
            # nrm = sparse.linalg.norm(next_step[start] - prev_step[start])
            nrm = norm()
            if info:
                print('step: ', i, '#elements: ', next_step[start].count_nonzero(), 'norm:', nrm)

            if nrm < tol:
                break

            i = i + 1
            for ind in self.grammar.nonterminals.values():
                prev_step[ind] = next_step[ind].copy()

        print(i)

        return prev_step

    # def nk_equation(self, step):
    #     step_matrices = dict()
    #
    #     for ind in self.grammar.terminals.values():
    #         step_matrices[ind] = self.matrices[ind]
    #
    #     for i, ind in enumerate(self.grammar.nonterminals.values()):
    #         step_matrices[ind] = step[i * self.en:(i + 1) * self.en, :]
    #
    #     step_matrices[self.alp] = sparse.csr_matrix((self.en, self.en))
    #
    #     for l in self.rules.keys():
    #
    #         r = self.rules[l]
    #         for r_i in r:
    #             if len(r_i) == 2:
    #                 step_matrices[self.alp] = step_matrices[self.alp] + r_i[1] * step_matrices[r_i[0]]
    #             else:
    #                 step_matrices[self.alp] = step_matrices[self.alp] + r_i[2] * step_matrices[r_i[0]].dot(step_matrices[r_i[1]])
    #         step_matrices[l] = step_matrices[self.alp]
    #         step_matrices[self.alp] = sparse.csr_matrix((self.en, self.en))
    #
    #     for i, ind in enumerate(self.grammar.nonterminals.values()):
    #         step[i * self.en:(i + 1) * self.en, :] = step_matrices[ind] - step[i * self.en:(i + 1) * self.en, :]
    #
    #     # start = self.grammar.nonterminals['S']
    #     # res_S = sparse.csr_matrix((self.en, self.en))
    #     # res_S[start * self.en:(start + 1) * self.en, :] = step_matrices[start] - step[start * self.en:(start + 1) * self.en, :]
    #
    #     return step

    def newton_krylov(self, equation, initial_guess=None, tol=10e-15, info=True):

        k = 1 #self.grammar.ntc

        if initial_guess is None:
            init_step = sparse.csr_matrix((self.en, self.en))
        else:
            init_step = initial_guess

        # if equation is None:
        #     equation = self.nk_equation

        res = nonlin_solve(equation, init_step, self.en, k, verbose=info, f_tol=tol)

        # start = self.grammar.nonterminals['S']
        # res_S = sparse.csr_matrix((self.en, self.en))
        # res_S[start * self.en:(start + 1) * self.en, :] = res[start * self.en:(start + 1) * self.en, :]

        return res
