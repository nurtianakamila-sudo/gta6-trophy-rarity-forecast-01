# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/nurtianakamila-sudo/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/nurtianakamila-sudo/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one: for each of the game's 55 trophies, five quantiles (5% / 25% / 50% / 75% / 95%) of its Exophase earn rate at the case target. A deterministic, offline, stdlib-only Python program. An earn rate is a ratio, earners over tracked players, and Exophase refreshes its counts hours apart, so the model treats the two regimes the data actually shows: when the history moves, a logit-linear trend in log time, extrapolated to the target with the move damped to 70% (trends decelerate as the cohort ages); when the history is frozen, the page is stale and a refresh is overdue, so the forecast dilutes every rate except 100% — at the first launch-night refresh the tracked-player count rose 6.06x while non-100% earners rose only 1.34x, so the average rate fell to about 0.22x. Practice score about 0.21–0.25 against the house persistence baseline's 1.51.

## Run it

```sh
MAKEFUN_INPUT_DIR=kit/input MAKEFUN_OUTPUT_DIR=out python3 repo/main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=09jncemn&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
