# iCEM attribution

The policy-guided optimizer in `src/cars/planning/sequence_optimizer.py` is a local
reimplementation/adaptation of algorithmic ideas from Pinneri et al., *Sample-efficient
Cross-Entropy Method for Real-time Planning*, CoRL 2020 / PMLR 2021.

- [Paper](https://proceedings.mlr.press/v155/pinneri21a.html)
- [Official source, pinned revision](https://github.com/martius-lab/iCEM/tree/98c1c1fe2bfc94a87e4658b6c793f4e81bc30203)
- [Upstream MIT license](LICENSE), copyright 2020 Autonomous Learning Group
- [Local source audit](../../docs/SEARCH_OPTIMIZER_AUDIT.md)

We optimize residuals around a recurrent feedback policy, with a fixed total
trajectory budget and explicit finite-DFT noise normalization. This is not an
unchanged upstream implementation, original-task reproduction or new optimizer
claim. Exact differences and measurements belong in the experiment report.
