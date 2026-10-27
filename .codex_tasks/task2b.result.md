Implemented both options for PPO/SAC, including automatic restoration during evaluation and defaults for old configs. Reward, termination, physics, metrics, and per-frame logging remain unchanged. Updated [docs/TASK2.md](/home/sasha/Desktop/cars_project/docs/TASK2.md).

- Tests: **48 passed**.
- PPO smoke: four envs, repeat 4, throttle-axis enabled; **50,176 decisions / 200,704 frames**.
- Seen-10: return **277.27 ± 116.92**, **0/10 finished**, no crashes.
- Verified all logs contain 1000 frame rows with four-frame action holds.

Active runs were untouched. No unseen/hard access, long training, or commit.