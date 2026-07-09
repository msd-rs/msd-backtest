---
name: optimize_strategy
description: guide user to optimize a strategy for msd backtest
tags: [msd, strategy, backtest, optimize]
version: 1.0
author: elsejj
---


# Workflow

1. Follow the user request to understand the strategy idea, write a initial basic strategy.
2. Do backtest
3. Review the metrics, analyze the shortcomings and make improvement suggestions.
4. If the strategy is not satisfied, start a new round (increment ${round_id}), follow the suggestions to improve and write strategy again, and goto 2. 
5. After optimize, write a full report

# Notes

- Don't create plan before review, all optimize steps are guided by backtest results.
- If user had forgot to provide a symbol list, ask to provide one, then create or copy it to workspace's `symbols.txt`.
- All variable like `${var}` in command should be replaced with actual values.
- Don't read/write files out of the workspace, except reading `.env` file, or files user explicitly point out.

# Terms

- `workspace`: the sub-directory for this strategy under `optimize` folder. All files you created, saved to this folder. Reference it with `${workspace}`.
- `round_id`: the id of the round, starts from 1, increase by 1 each round, zero-padded to 2 digits, e.g. `01`, `02`. Reference it with `${round_id}`.
- `symbols`: a list of symbols to backtest. `symbols.txt` is the default file name, it can be changed by user. It's placed in the workspace. Reference it with `${symbols}`. 
- `strategy_file`: The name of the strategy file, it should be `strategy.${round_id}.py`. Reference it with `${strategy_file}`.
- `backtest_output_file`: The name of the backtest output file, it should be `backtest.${round_id}.json`. Reference it with `${backtest_output_file}`.
- `report_file`: The name of the report file, it should be `report.${round_id}.md`. Reference it with `${report_file}`.

# Action details

## Write strategy 

1. Create a `strategy_file` for this round. Follow previous round's `report_file` to improve. 
2. You can freely experiment from multiple perspectives, add constraints, use more data, test different parameters, while keeping the user's core needs in mind.
3. Using `write-strategy` skill to write it.


## Do backtest

- Backtest the strategy with `uv run python -m mbt.run.cli ${strategy_file} -s ${symbols} -o ${backtest_output_file}`. This will be generate `backtest_output_file`, which contains detailed performance metrics for each symbol.
- You can use `uv run python -m mbt.run.cli --help` to get more information about the command.
- You can also edit `.env` to modify some fixed arguments


## Review the metrics 

- Because `backtest_output_file` may be very large, you should use `uv run python -m mbt.run.review ${backtest_output_file}` to explore the data instead reading the file directly. There are 3 sub command of `uv run python -m mbt.run.review ${backtest_output_file}`:
  - `stat`: `python -m mbt.run.review ${backtest_output_file} stat` to review the statistics of the metrics.
  - `detail`: `python -m mbt.run.review ${backtest_output_file} detail -s ${symbol}` to review the metrics of a specific symbol.
  - `sort`: `python -m mbt.run.review ${backtest_output_file} sort -m ${metric}` to show the metric sorted. the metric can be any of:
    - `profit` : the final profit of the strategy.
    - `profit_original` : the final profit during the test but without apply the strategy.
    - `alpha` : the alpha of the strategy.
    - `beta` : the beta of the strategy.
    - `sharpe` : the sharpe ratio of the strategy.
    - `information_ratio` : the information ratio of the strategy.
    - `ir` : the information ratio of the strategy.
    - `sortino` : the sortino ratio of the strategy.
    - `treynor` : the treynor ratio of the strategy.
    - `max_drawdown` : the maximum drawdown of the strategy.
    - `calmar` : the calmar ratio of the strategy.
    - `win_rate` : the win rate of the strategy.
    - `avg_win` : the average win of the strategy.
    - `avg_loss` : the average loss of the strategy.
    - `profit_factor` : the profit factor of the strategy.
    - `fee_rate` : the fee rate of the strategy.
    - `trade_count` : the trade count of the strategy.
- All review result are JSON format and printed to stdout.
- You may run the review commands multiple times to explore the data.
- Write a report to analyze the metrics to `report.${round_id}.md` using chinese, mention the key findings and insights.
- Point out the shortcomings and suggests to improve.
- Use the report, rewrite a strategy file for next round

## Full Report

- Use chinese when writing the full report, write it to `report.md`.
- The report should include the following sections:
  - Strategy Introduction
  - Backtest results summary, with top 10 better and worse symbols.
  - Optimization process and key insights
  - Explanation of advantages
  - Reflection on the shortcomings and limitations
  - Conclusion and outlook
