# optimneuralbandits
Neural Bandits with an optimizer to generate a set of contexts to evaluate at every turn (possibly other optimizers, the code is pretty agnostic in that regard), to generate candidate vectors to evaluate/pull

Original versions of NeuralUCB / TS that are modified here: https://github.com/uclaml/NeuralTS

Includes:
* Vanilla NeuralTS and NeuralUCB
* NeuralTS with Dropout TS
* NeuralTS with Lenient Regret


Relevant papers:
* Zhang, Weitong, et al. "Neural thompson sampling." arXiv preprint arXiv:2010.00827 (2020).
* Zhou, Dongruo, Lihong Li, and Quanquan Gu. "Neural contextual bandits with ucb-based exploration." International Conference on Machine Learning. PMLR, 2020.
* Riquelme, Carlos, George Tucker, and Jasper Snoek. "Deep bayesian bandits showdown: An empirical comparison of bayesian deep networks for thompson sampling." arXiv preprint arXiv:1802.09127 (2018).
* Merlis, Nadav, and Shie Mannor. "Lenient regret for multi-armed bandits." arXiv preprint arXiv:2008.03959 (2020).
