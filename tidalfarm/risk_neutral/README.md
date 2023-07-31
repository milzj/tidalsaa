# Risk-neutral design optimization of tidal-stream energy farms

Motivated by ["Uncertain bottom friction and viscosity: A case study"](https://github.com/milzj/tidalsaa/tree/nominal/tidalfarm/nominal#uncertain-bottom-friction-and-viscosity-a-case-study),
we consider the risk-neutral desgin of tidal-streem energy farms 

$$
	\min_{u \in U_{\text{ad}}}  \mathbb{E}[J(S(u,\xi)),u)] + \beta \\|u\\|_{L^1(D)},
$$

where $J$, $\beta$, $S(u,\xi)$ are as described in ["Optimization problem"](https://github.com/milzj/tidalsaa/tree/nominal/tidalfarm/nominal#optimization-problem), but the notation $S(u,\xi)$
instead of $S(u)$ hightlightes dependence on the simulation output on parameters $\xi$, such as bottom friction. 


![](https://github.com/milzj/tidalsaa/blob/nominal/tidalfarm/nominal/output/10-May-2023-13-44-27_solution_best_n%3D100_online_version.png)
|:--:| 
*Nominal optimal turbine density (with fixed bottom friction)*
![](https://github.com/milzj/tidalsaa/blob/nominal/tidalfarm/nominal/output/10-May-2023-13-44-27_solution_best_n%3D100_online_version.png)
|:--:| 
*Risk-neutral optimal turbine density (with uncertain bottom friction)*
