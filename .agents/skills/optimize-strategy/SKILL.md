---
name: optimize_strategy
description: guide user to optimize a strategy for msd backtest
tags: [msd, strategy, backtest, optimize]
version: 1.0
author: elsejj
---


# Workflow

1. Write strategy 
2. Do backtest
3. Review the metrics.
4. Goto 1 until satisfied or your had try 5 times.


# Action details

## Write strategy 

1. Create a sub-directory for this strategy under `mbt/strategy` folder. The sub-directory is your workspace, all files you created, saved to this folder.
2. Append `round-$N` to the suffix of each file name, e.g. `strategy.py` -> `strategy_round1.py`, and make a copy of `round-1` to `round-2` and so on. `N` is the round number, starts from 1.
3. Follow the user requirement to write a strategy using `write-stra` skill.


## Do backtest

When you had written a strategy use `write-stra` skill, following command is your can use to optimize it


- Create a `symbols.txt` file, and append the symbols you want to backtest to it. The symobl format is `<market><instrument_id>`, e.g. 'SH600519' or 'SZ000001', each symbol is on a new line. Up to 300 symbols allowed in a single backtest.

- Backatest the strategy with `uv run python -m mbt.run.cli <your strategy file> -s <symbols file>`, After backtest, it report some metrics like alpha, beta ... to stdio with a JSON array, each record's `symbol` property's value is the symbol,  other fields are `metrics` which is a float value and its key is the metric name.  Use it to decide whether the strategy is meet the needs.
- When you need to review the detail of the trades in a backtest, use `uv run python -m mbt.run.cli <your strategy file> -s <your symbol 1> <your symbol 2> ... <your symbol n> -o <output_file>`, output to a JSON file will be more convenient to review, the output file may be too large, so this step is only suggest when you need to dive deep to the result.
- Save the outputs of the backtest to a file, the name should be like `backtest_round$N.json`.
- You can use `uv run python -m mbt.run.cli --help` to get more information about the command.
- You can also edit `.env` to modify some fixed arguments


## Review the metrics 

- Write a report to analyze the metrics to `report_round$N.md` using chinese, mention the key findings and insights.
- Point out the shortcomings and suggests to improve.
- Use the report, rewrite a strategy file for next round