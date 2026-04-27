# Probability-Driven Statistical Modeling and Verification of Random Signals and Noise in Electrical Power Systems

This repository contains the simulation and verification workflow for a probability-based research project on random signals and noise in electrical power systems.

The project models a measured electrical power waveform as a combination of:

- A nominal 50 Hz fundamental component
- Harmonic distortion
- Decaying transient disturbance
- Slow stochastic load and renewable-energy variations
- Additive Gaussian measurement noise

The main goal is to show how probability and statistics can be used as engineering tools to quantify uncertainty, measurement accuracy, signal distortion, and transient-detection reliability in modern power systems.

---

## Research Motivation

Electrical power-system measurements are never perfectly deterministic. Real voltage and current signals are affected by:

- Sensor noise
- Harmonic distortion
- Switching transients
- Random load behavior
- Renewable-energy intermittency
- Measurement and communication imperfections

Because protection, monitoring, control, and estimation algorithms depend on measured waveforms, it is important to model uncertainty mathematically instead of treating noise as an unwanted side effect.

This project builds a stochastic signal model and verifies probability-based monitoring formulas using Monte Carlo simulations.

---

## Main Research Idea

The measured waveform is modeled as:

```math
x[k] = s[k] + h[k] + \tau[k] + \delta[k] + n[k]
