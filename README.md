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
<table>
    <thead>
    <tr>
      <th rowspan="2" align="left">Graph</th>
      <th rowspan="2" align="left">V</th>
      <th rowspan="2" align="left">E</th>
      <th rowspan="2" align="left">nnz</th>
      <th colspan="2" align="center">naive-iteration</sub></th>
      <th colspan="2" align="center">newton-krylov</sub></th>
    </tr>
    <tr>
        <td>it</td>
        <td>time</td>
        <td>it</td>
        <td>time</td>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>generations</td>
        <td>129</td>
        <td>273</td>
        <td>2164</td>
        <td>5</td>
        <td>0.02</td>
        <td>2</td>
        <td>0.4</td>
    </tr>
    <tr>
        <td>travel</td>
        <td>131</td>
        <td>277</td>
        <td>2499</td>
        <td>11</td>
        <td>0.05</td>
        <td>3</td>
        <td>0.6</td>
    </tr>
    <tr>
        <td>funding</td>
        <td>778</td>
        <td>1086</td>
        <td>17634</td>
        <td>9</td>
        <td>0.04</td>
        <td>3</td>
        <td>0.7</td>
    </tr>
    <tr>
        <td>wine</td>
        <td>733</td>
        <td>1839</td>
        <td>66572</td>
        <td>11</td>
        <td>0.06</td>
        <td>3</td>
        <td>0.7</td>
    </tr>
    <tr>
        <td>pizza</td>
        <td>671</td>
        <td>1980</td>
        <td>15195</td>
        <td>15</td>
        <td>0.09</td>
        <td>4</td>
        <td>0.9</td>
    </tr>
    <tr>
        <td>core</td>
        <td>1323</td>
        <td>2752</td>
        <td>204</td>
        <td>50</td>
        <td>0.3</td>
        <td>4</td>
        <td>0.9</td>
    </tr>
    <tr>
        <td>pathways</td>
        <td>6238</td>
        <td>12363</td>
        <td>884</td>
        <td>9</td>
        <td>0.2</td>
        <td>4</td>
        <td>2.1</td>
    </tr>
    <tr>
        <td>enzyme</td>
        <td>48815</td>
        <td>86543</td>
        <td>396</td>
        <td>9</td>
        <td>0.2</td>
        <td>4</td>
        <td>5.2</td>
    </tr>
    <tr>
        <td>eclass</td>
        <td>239111</td>
        <td>360248</td>
        <td>90994</td>
        <td>11</td>
        <td>19.6</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>go-hierarchy</td>
        <td>45007</td>
        <td>490109</td>
        <td>588976</td>
        <td>17</td>
        <td>8.3</td>
        <td>10</td>
        <td>144.2</td>
    </tr>
    <tr>
        <td>geospecies</td>
        <td>450609</td>
        <td>2201532</td>
        <td>91</td>
        <td>3</td>
        <td>0.2</td>
        <td>3</td>
        <td>9.6</td>
    </tr>
  </tbody>
</table>


**g<sub>2</sub>**
```
S -> subClassOf_r S subClassOf | subClassOf
```

<table>
    <thead>
    <tr>
      <th rowspan="2" align="left">Graph</th>
      <th rowspan="2" align="left">V</th>
      <th rowspan="2" align="left">E</th>
      <th rowspan="2" align="left">nnz</th>
      <th colspan="2" align="center">naive-iteration</sub></th>
      <th colspan="2" align="center">newton-krylov</sub></th>
    </tr>
    <tr>
        <td>it</td>
        <td>time</td>
        <td>it</td>
        <td>time</td>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>generations</td>
        <td>129</td>
        <td>273</td>
        <td>0</td>
        <td>0</td>
        <td>&lt;0.01</td>
        <td>0</td>
        <td>&lt;0.01</td>
    </tr>
    <tr>
        <td>travel</td>
        <td>131</td>
        <td>277</td>
        <td>63</td>
        <td>7</td>
        <td>0.02</td>
        <td>2</td>
        <td>0.3</td>
    </tr>
    <tr>
        <td>funding</td>
        <td>778</td>
        <td>1086</td>
        <td>1158</td>
        <td>7</td>
        <td>0.02</td>
        <td>2</td>
        <td>0.4</td>
    </tr>
    <tr>
        <td>wine</td>
        <td>733</td>
        <td>1839</td>
        <td>133</td>
        <td>7</td>
        <td>0.02</td>
        <td>2</td>
        <td>0.3</td>
    </tr>
    <tr>
        <td>pizza</td>
        <td>671</td>
        <td>1980</td>
        <td>1262</td>
        <td>13</td>
        <td>0.04</td>
        <td>3</td>
        <td>0.5</td>
    </tr>
    <tr>
        <td>core</td>
        <td>1323</td>
        <td>2752</td>
        <td>214</td>
        <td>7</td>
        <td>0.02</td>
        <td>2</td>
        <td>0.3</td>
    </tr>
    <tr>
        <td>pathways</td>
        <td>6238</td>
        <td>12363</td>
        <td>3117</td>
        <td>9</td>
        <td>0.02</td>
        <td>4</td>
        <td>0.8</td>
    </tr>
    <tr>
        <td>enzyme</td>
        <td>48815</td>
        <td>86543</td>
        <td>8163</td>
        <td>9</td>
        <td>0.05</td>
        <td>6</td>
        <td>1.5</td>
    </tr>
    <tr>
        <td>eclass</td>
        <td>239111</td>
        <td>360248</td>
        <td>96163</td>
        <td>9</td>
        <td>5.3</td>
        <td>5</td>
        <td>213</td>
    </tr>
    <tr>
        <td>go-hierarchy</td>
        <td>45007</td>
        <td>490109</td>
        <td>738937</td>
        <td>17</td>
        <td>3.6</td>
        <td>9</td>
        <td>77.0</td>
    </tr>
    <tr>
        <td>geospecies</td>
        <td>450609</td>
        <td>2201532</td>
        <td>0</td>
        <td>0</td>
        <td>&lt;0.01</td>
        <td>0</td>
        <td>&lt;0.01</td>
    </tr>
  </tbody>
</table>
