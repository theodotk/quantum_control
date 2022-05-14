# Two-level system excitation

This project is an easy example of quantum control with reinforcement learning. It contains:
+ the visualisation of the result is in [demo_control.ipynb](https://colab.research.google.com/drive/1PEYf3yy9L-YBHH-lfhnlfr82caW4kRYZ),
+ a trained model,
+ the code for training is in [training.py](??) and can be launched as `python3 training.py --n-trials NUM_OF_OPTIMIZATION_TRIALS --name NAME_OF_THE_SAVED_MODEL`

## The problem from physical perspective

TLDR: it's like an inverted pendulum, but fancy

### The state

We describe a quantum system that has two energy eigenstates:
+ ground state (with low energy): $|g\rangle$
+ excited state (with high energy): $|e\rangle$

A quantum system can be in any combination of the eigenstates, for example:

$|p\rangle = \sqrt{1-P_e}\, |g\rangle + \alpha \sqrt{P_e}\, |e\rangle$.

Here $\alpha$ is a complex number satisfying $|\alpha| = 1$, and $P_e$ is a real number between 0 and 1: the probability to find the system in $|e\rangle$. It is also called the excited state population.

### Bloch sphere

Let's imagine, what it looks like. Parameter $P_e$ quantifies the amount of $|e\rangle$ in the current state. Thus it represents an axis moving along which one travels between $|e\rangle$ and $|g\rangle$.

If $P_e$ is equal to 1 or 0, we are in one of the eigenstates. The parameter $\alpha$ does not matter in this case, because in quantum states we don't care about the global phase [^ phase].

For other values of $P_e$ the phase matters, and is actually the core of all the fun stuff in quantum pysics. But let's recall that it is a complex number of module 1. All such numbers form a circle of radius 1, with the coorditates $x = Re({\alpha})$ and $y = Im({\alpha})$. And thus $\alpha \sqrt{P_e}$ is a circle of radius $\sqrt{P_e}$.

It means we can map any state of type $|p\rangle$ onto the surface of a shpere, where each point has coordinates:
- $x = 2\sqrt{P_e(1-P_e)} Re({\alpha})$
- $y = 2\sqrt{P_e(1-P_e)} Im({\alpha})$
- $z = 2P_e - 1$

That is the case only for the pure states. The state with time loses its purity due to incoherent processes and the $(x,y,z)$ move inside the sphere.

Thus any dynamics of a two level system can be represented as a movement on the surface and inside a raduis 1 sphere. Moving along the Z axis corresponds to change in the population of the excited and ground states, while the movement in a plane $z=$const corresponds to the change of the phase.

### Moving between the states

Putting it simple, within this problem, there are two ways for the system to move between the ground and excited states [^ 1]:
- Excitatation by an electromagnetic pulse (in following I call it a laser because I'm used to the optical context, but it depends on a system). It works both ways: the system in the ground state will gain some energy and move closer to the excited state, while a system in the excited state will be pushed to emit some energy and will move towards the ground state. On a sphere it can be representd as a rotation around an axis that is situated in the x-y plane, with frequency defined by laser power (and other factors we don't model here like coupling efficiency). 
- Spontaneous emission: the system emits a photon, thus going down from the excited state to the ground state with the rate $\gamma$. If we start in the excited state, the trajectory on a sphere will coincide with Z axis.

The laser can have a frequency different from the resonance frequency of the TLS, the difference is called detuning $\delta$. This results in the rotation around the Z axis with frequency $\delta$.

Other incoherent processes result in increase of our unawareness of the stae of the system, this results in dephasing witrh rate $\gamma^*$.

### Quantum control

In general it's the field studying the optimal conditions to perform a given quantum operation with maximal fidelity. A usual tool is shaping of the control pulse (providing pulse power and phase during given time). There are different algorithms tailored for this purpose (GRAPE, Krotov, GOAT, CRAB etc), but it's also possible to use reinforcement learning (RL) to solve this problem.

The task in our case is:
- to put the TLS into the excited state
- starting from any pure state
- with arbitrary conditions (detuning $\delta$, spontaneous emission rate $\gamma$ and dephasing rate $\gamma^*$)

It is a rather straightforward task with an [analytical solution](https://en.wikipedia.org/wiki/Rabi_problem#Two-level_atom), but let's look at the cases and on the variables we need for an RL setting.

1. If $\delta = \gamma = \gamma^* = 0$, we just need to drive the system with laser until it reaches the excited state, and then stop. It would require to have one observable ($z$ coordinate) and one action (laser power).
1. With non-zero dephasing the state rotates around the Z axis and thus we also need to adjust the phase of the laser as $\phi = \delta t$. The phase will be the second action, and we'll need another coordinate in the observables so that the model can calculate $\delta$.
1. If $\gamma > 0$ the system will head for the ground state from all the states. The model will need to adjust the power, and also if the purpose is to be as close to the excited state for as long as possible, there may appear different strategies to counter the spontaneous emission (like going back and forth to the ground state if there is a particular relation between the variables).


[^ phase]: we need the states only for calculation of the observables, that involves taking absolute value. So the global phase will just not be measureable, and thus we don't care.
[^ 1]: essentially, both are the same. Tt is interaction with an electromagnetic bath.


##  The problem from the ML perspective

It looks like a quantum analogue of the common problem of inverting a pendulum: one needs to fix a system in a state with the higher energy in presence of the field pushing it towards the lower energy state.

|System|Pendulum|TLS excitation|
|-|-|-|
|Coordinate|Angle with respect to the upper position|Probability to find the system in an excited state|
|Force|gravity ($g$)|Spontaneous emission ($\gamma$)|
|Action|Torque (2nd derivative of coordinate)|Laser excitation (adds to 1st derivative of coordinate)|

The difference is that the pendulum has a fixed length, so the gravity can be coutereb by the torque, while a TLS' state is decaying towards ground state no matter what, like a pendulum on a rope can just fall down instead of rotating.

To make it a little more interesting, I added the following:
- It's a 2d problem: one can also choose the rotation axis
- There is a force rotating the TLS state around Z axis (analytically we can easily get rid of it, but then the problem will be simpler)
- My goal was to create a model that will adapt to different conditions rather than will be tailored for one set of arameters

### Model

#### Observables

I choose the $x$, $y$, $z$ coordinates, as they give information both on the state position and purity (I could have chosen the two angles and the purity as well). Due to the experiments with different targets, I also introduced two angles between the current and the target state. Different targets was too much for the model to handle, so this will be the next step.

#### Actions

There are two actions: laser power and phase. In a normal setting the power would be enough, and rotation of the state due to the detuning can be handled analytically, but my goal was making an adaptative model, and it needed an extra parameter.

#### Reward

In this case there were two strategies:
- to place the system in the target state and then to abort the simulation
- to try to keep it there or as close as possible

I went with the secod one, and thus the reward is a fidelity of the system state being the target state (basically, an analogue of the scalar product between vectors).

#### The model

I experimented with the pendulum, and it seems that SAC grants the best results in terms of speed and performance. I tried it with optuna optimization on a stationary TLS excitation problem, and than on this one, and I'm rather satisfied by the result. The PPO also gives similar performance, but requires more time to train.

Because I want the model to adapt to different parameters, I trained and evaluated models on a set of randomly generated environements.

# Next steps

- going to any target
- state transfer in a three-level system (STIRAP)
- entangled state creation
