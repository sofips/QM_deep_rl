# QM_deep_RL

Implementation of the algorithm developed in the work of Zhang et. al (https://doi.org/10.1103/PhysRevA.97.052333). 
The following
code uses Deep Reinforcement Learning to obtain an optimal sequence of magnetic fields to be applied at the extremes of a spin chain in order to achieve perfect transmission of a quantum state.

## Physical system

We consider an homogeneous XX hamiltonian describing a quantum spin chain that can be under the influence of magnetic pulses in discrete periods of time:

$$
\mathbf{H( t)  = -\sum _{i=1}^{N-1} J\left( \sigma _{i}^{x} \sigma _{i+1}^{x} +\sigma _{i}^{y} \sigma _{i+1}^{y}\right) +\sum _{i=1}^{N} B_{k}( t) \sigma _{k}^{z}}
$$  

Following the work of Zhang et. al, consider 16 possible combinations of pulses that can be applied to the extremes of the chain. 

## Implementation

The DRL algorithm is implemented both in Pytorch and keras. Keras provides a simpler structure which can be easier to understand, while Pytorch allows GPU implementation. 

A full description can be found in https://doi.org/10.1103/PhysRevA.97.052333
