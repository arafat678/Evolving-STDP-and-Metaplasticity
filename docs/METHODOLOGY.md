# Methodology: Evolving STDP, Metaplasticity, and Homeostasis in SNNs

## 1. Theoretical Framework
This study investigates how biologically plausible, fully local synaptic plasticity rules can be evolved to solve the fundamental **stability-plasticity dilemma** in Spiking Neural Networks (SNNs) operating under sequential, task-incremental continual learning scenarios.

---

## 2. Spiking Neuron & Synaptic Dynamics

### 2.1 Leaky Integrate-and-Fire (LIF) Dynamics
A recurrent spiking network with $N_{\text{in}} = 64$ Poisson input neurons, $N_h = 24$ excitatory hidden neurons, and lateral inhibition $w_{\text{inh}}$ is simulated across discrete time steps $t \in [1, K]$ ($K = 20$):
$$V_i(t) = \beta V_i(t-1) + \sum_j W_{ij}(t-1) x_j(t) - w_{\text{inh}} \sum_{k \ne i} S_k(t-1)$$
where a spike is triggered when membrane potential exceeds dynamic threshold:
$$S_i(t) = \Theta(V_i(t) - \theta_i(t))$$

### 2.2 Adaptive Threshold Homeostasis
To prevent runaway excitation and maintain target firing rates:
$$\theta_i(t) = \theta_0 + \gamma \cdot \text{trace}_i(t)$$
$$\tau_{\text{th}} \frac{d\,\text{trace}_i}{dt} = - \text{trace}_i(t) + S_i(t)$$

---

## 3. Synaptic Plasticity & Metaplasticity Formulations

### 3.1 Trace-Based Spike-Timing-Dependent Plasticity (STDP)
Pre-synaptic and post-synaptic spike traces are updated as:
$$x_j^{\text{pre}}(t) = x_j^{\text{pre}}(t-1) e^{-dt / \tau_p} + S_j^{\text{pre}}(t)$$
$$x_i^{\text{post}}(t) = x_i^{\text{post}}(t-1) e^{-dt / \tau_m} + S_i^{\text{post}}(t)$$

Standard pair-based potentiation and depression:
$$\Delta W_{ij}^{\text{STDP}}(t) = A_+ x_j^{\text{pre}}(t) S_i^{\text{post}}(t) - A_- x_i^{\text{post}}(t) S_j^{\text{pre}}(t)$$
where the depression-to-potentiation ratio is parameterized by $\rho = A_- / A_+$.

### 3.2 Metaplasticity Gate Modulation (G1 & G2)
Metaplasticity modifies the susceptibility of a synapse to subsequent plasticity based on past history of activity (synaptic consolidation):
- **Gate G1:**
  $$\Delta W_{ij}(t) = \frac{\Delta W_{ij}^{\text{STDP}}(t)}{1 + m \cdot C_{ij}(t)}$$
  where $C_{ij}(t)$ tracks cumulative synaptic weight transitions, shielding stabilized synapses from catastrophic forgetting during future tasks.
- **Gate G2:**
  $$\Delta W_{ij}(t) = \Delta W_{ij}^{\text{STDP}}(t) \cdot \exp\left(-\frac{C_{ij}(t)}{c_0}\right)$$

---

## 4. Multi-Objective Evolutionary Optimization

The 10-dimensional parameter genome:
$$\mathbf{z} = [A_+, \rho, \tau_p, \tau_m, \text{meta}, \eta, \theta_0, w_{\text{inh}}, \gamma, \tau_{\text{th}}]$$
is optimized to maximize both continual learning objectives simultaneously:
1. **Forward Plasticity ($f_{\text{pl}}$):** Peak task-acquisition accuracy on new classes.
2. **Backward Stability ($f_{\text{st}}$):** Retention accuracy on previously mastered classes without retraining.

Optimizers compared:
- **NSGA-II:** Non-dominated Sorting Genetic Algorithm II (crowding distance diversity).
- **CMA-ES:** Covariance Matrix Adaptation Evolution Strategy.
- **TPE-MO:** Multi-Objective Tree-structured Parzen Estimator.
- **Random Search:** Uniform prior baseline.
- **Hand-Tuned (Diehl & Cook style):** Heuristic baseline with no metaplasticity.
