# Stochastic Context-Free Path Querying

This is the implementation of several methods to solve the most probable paths problem of Stochastic context-free path querying (SCFPQ). **SCFPQ** --- a way to specify path constraints in terms of stochastic context-free grammars. Our main task here to solve most probable path problem. **Most probable paths problem** is for a given stochastic context-free grammars and a labeled directed graph, to find the maximum probabilities between all pairs of nodes.

Our solution is a reduction to systems of nonlinear matrix equation solving. Two methods to solve this systems is proposed:

- Naive Iterative Method
- Newton-Krylov Method


## Requirements 

SCFPQ requires the following to run:
- Python v3.7+
- CUDA Toolkit 10.2+
- cupy 

## Usage

```
from main_structures import Grammar, Graph, Equation
```
You can download grammar and graph from file:
```
grm = Grammar(grammars_path)
grh = Graph(graph_path)
grh.fill()
```
And then construct the equation with:
```
eq = Equation(grm, grh)
```
Now you can solve the obtained system with auto-constracted function with two different methods:
- Naive Iterative Method
    ```
    res = eq.naive_iteration()
    ```
- Newton-Krylov Method
    ```
    res = eq.newton_krylov()
    ```

Or you can also choose your function, initial guess for iterative process and tolerance:
```
res = eq.newton_krylov(equation=another_function, initial_guess=x0, tol=10e-6, info=True)
```

## Dataset

The graph data is selected from CFPQ_Data dataset. Graphs related to RDF analysis problems was chosen.

## Performance

The results of the most probable paths problem on graphs related to RDF analysis are listed below.

**g<sub>1</sub>**
```
S -> subClassOf_r S subClassOf | subClassOf_r subClassOf 
     | type_r S type | type_r type
```

| Graph        |   V    |    E    |  nnz   | it |  time  | it | time  |
|--------------|--------|---------|--------|----|--------|----|-------|
| generations  | 129    | 273     | 2164   | 5  | 0.02   | 2  | 0.4   |
| travel       | 131    | 277     | 2499   | 11 | 0.05   | 3  | 0.6   |
| funding      | 778    | 1086    | 17634  | 9  | 0.04   | 3  | 0.7   |
| wine         | 733    | 1839    | 66572  | 11 | 0.06   | 3  | 0.7   |
| pizza        | 671    | 1980    | 15195  | 15 | 0.09   | 4  | 0.9   |
| core         | 1323   | 2752    | 204    | 50 | 0.3    | 4  | 0.9   |
| pathways     | 6238   | 12363   | 884    | 9  | 0.2    | 4  | 2.1   |
| enzyme       | 48815  | 86543   | 396    | 9  | 0.2    | 4  | 5.2   |
| eclass       | 239111 | 360248  | 90994  | 11 | 19.6   | -  | -     |
| go-hierarchy | 45007  | 490109  | 588976 | 17 | 8.3    | 10 | 144.2 |
| geospecies   | 450609 | 2201532 | 91     | 3  | 0.2    | 3  | 9.6   |


**g<sub>2</sub>**
```
S -> subClassOf_r S subClassOf | subClassOf
```

| Graph        |   V    |    E    |  nnz   | it |  time | it | time  |
|--------------|--------|---------|--------|----|-------|----|-------|
| generations  | 129    | 273     | 0      | 0  | <0.01 | 0  | <0.01 |
| travel       | 131    | 277     | 63     | 7  | 0.02  | 2  | 0.3   |
| funding      | 778    | 1086    | 1158   | 7  | 0.02  | 2  | 0.4   |
| wine         | 733    | 1839    | 133    | 7  | 0.02  | 2  | 0.3   |
| pizza        | 671    | 1980    | 1262   | 13 | 0.04  | 3  | 0.5   |
| core         | 1323   | 2752    | 214    | 7  | 0.02  | 2  | 0.3   |
| pathways     | 6238   | 12363   | 3117   | 9  | 0.02  | 4  | 0.8   |
| enzyme       | 48815  | 86543   | 8163   | 9  | 0.05  | 6  | 1.5   |
| eclass       | 239111 | 360248  | 96163  | 9  | 5.3   | 5  | 213   |
| go-hierarchy | 45007  | 490109  | 738937 | 17 | 3.6   | 9  | 77.0  |
| geospecies   | 450609 | 2201532 | 0      | 0  | <0.01 | 0  | <0.01 |
