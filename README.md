# Enhanced Colliding Bodies Optimization

This is a pythonic implementation of ECOB, an optimization algorithm developed by A. Kaveh and M.Ilchi Ghazaan.
[Link to the paper.](https://www.sciencedirect.com/science/article/abs/pii/S096599781400129X#:~:text=CBO%20is%20a%20population%2Dbased,governing%20laws%20from%20the%20physics.)

<br>
Colliding Bodies Optimization (CBO) is a new multi-agent meta-heuristic optimization algorithm based on
the physical laws governing one-dimensional collision between two bodies. Each agent is modelled as a 
body with a specified mass and velocity. A collision occurs between pairs of objects and new positions 
are updated to find the global or near-global solutions. Enhanced colliding bodies optimization (ECBO) 
uses memory to save the best solutions and utilizes a mechanism to escape local optima. The code is presented in Python as a jupyter notebook as well as a python file.

<br>
Following images show the distribution of the colliding bodies in three dimensions when ECOB is applied to minimize the function:<br>

```
(x-20)**2 + (y - 20)**2 = 0
```

<h2>Iteration 0</h2>

![Iteration 0](https://github.com/Cossak/Enhanced-Colliding-Bodies-Optimization/blob/main/iter0.png "Particles at iteration 0")

<h2>Iteration 20</h2>

![Iteration 0](https://github.com/Cossak/Enhanced-Colliding-Bodies-Optimization/blob/main/iter20.png "Particles at iteration 0")

<h2>Iteration 40</h2>

![Iteration 0](https://github.com/Cossak/Enhanced-Colliding-Bodies-Optimization/blob/main/iter40.png "Particles at iteration 0")

```
print("Best answer:", np.round(cb[0], 2))
Best answer: [ 0.07 20.   20.  ]
```
Correct answer: [don't care, 20, 20]
