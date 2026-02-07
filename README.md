# E2E-Signal-Temporal-Logic-Planner

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Conference](https://img.shields.io/badge/ICRA-2026-blue)

**Structured-MoE STL Planner (S-MSP)**: an end-to-end differentiable framework that maps synchronized multi-view camera observations and an STL specification directly to a feasible trajectory.

> 🎉 **Accepted to ICRA 2026**.

---

## Teaser

<p align="center">
  <img src="assets/teaser.png" width="92%" />
</p>
<p align="center">
  <em>Teaser: S-MSP maps multi-view observations and STL specifications directly to feasible trajectories with a structure-aware MoE and a safety filter.</em>
</p>

---

## News
- **2026-__-__**: Paper accepted to **ICRA 2026**.
- **2026-__-__**: Code release.

---

## Abstract

```latex
\begin{abstract}
We investigate the task and motion planning problem for Signal Temporal Logic (STL) specifications in robotics. Existing STL methods rely on pre-defined maps or mobility representations, which are ineffective in unstructured real-world environments.
We propose the \emph{Structured-MoE STL Planner} (\textbf{S-MSP}), a differentiable framework that maps synchronized multi-view camera observations and an STL specification directly to a feasible trajectory. S-MSP integrates STL constraints within a unified pipeline, trained with a composite loss that combines trajectory reconstruction and STL robustness. A \emph{structure-aware} Mixture-of-Experts (MoE) model enables horizon-aware specialization by projecting sub-tasks into temporally anchored embeddings.
We evaluate S-MSP using a high-fidelity simulation of factory-logistics scenarios with temporally constrained tasks. Experiments show that S-MSP outperforms single-expert baselines in STL satisfaction and trajectory feasibility. A rule-based \emph{safety filter} at inference improves physical executability without compromising logical correctness, showcasing the practicality of the approach.
\end{abstract}
