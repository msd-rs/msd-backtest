
# Workflow

1. Follow the user request to understand the strategy idea, write a initial basic strategy.
2. Do backtest
3. Review the metrics, analyze the shortcomings and make improvement suggestions.
4. If the strategy is not satisfied, start a new round (increment ${round_id}), follow the suggestions to improve and write strategy again, and goto 2. 
5. After optimize, write a full report

# Notes

- Don't create plan before review, all optimize steps are guided by backtest results.
- If user had forgot to provide a obj list or there is no `Selector`, ask to provide one, then create or copy it to workspace's `objs.txt`.
- All variable like `${var}` in command should be replaced with actual values.
- Don't read/write files out of the workspace, except reading `.env` file, or files user explicitly point out.

# Terms

- `workspace`: the sub-directory for this strategy under `optimize` folder. All files you created, saved to this folder. Reference it with `${workspace}`.
- `round_id`: the id of the round, starts from 1, increase by 1 each round, zero-padded to 2 digits, e.g. `01`, `02`. Reference it with `${round_id}`.
- `objs`: a list of objs to backtest. `objs.txt` is the default file name, it can be changed by user. It's placed in the workspace. Reference it with `${objs}`. 
- `strategy_file`: The name of the strategy file, it should be `strategy.${round_id}.py`. Reference it with `${strategy_file}`.
- `backtest_output_file`: The name of the backtest output file, it should be `backtest.${round_id}.json`. Reference it with `${backtest_output_file}`.
- `report_file`: The name of the report file, it should be `report.${round_id}.md`. Reference it with `${report_file}`.

# Action details

## Write strategy 

1. Create a `strategy_file` for this round. Follow previous round's `report_file` to improve. 
2. You can freely experiment from multiple perspectives, add constraints, use more data, test different parameters, while keeping the user's core needs in mind.

## Do backtest

- Backtest the strategy with `uv run python -m mbt.run.cli ${strategy_file} -s ${objs} -o ${backtest_output_file} -S` (if `${strategy_file}` already contains a `Selector` subclass, `-s ${objs}` can be omitted).
  - `-o ${backtest_output_file}` saves the full detailed report for each obj to disk.
  - `-S` (`--summary`) directly prints a compact **Markdown statistical summary** (including overview counts, distribution statistics across all metrics, and Top 10 / Bottom 10 objs by profit) to stdout, allowing you to immediately evaluate the round without reading the large JSON file.
- You can use `uv run python -m mbt.run.cli --help` to get more information about the command.
- You can also edit `.env` to modify some fixed arguments


## Review the metrics 

- Because `backtest_output_file` may be very large, never read the JSON file directly. Start from the Markdown summary printed by `-S` (or re-generate it via `uv run python -m mbt.run.review ${backtest_output_file} summary`), and use `uv run python -m mbt.run.review ${backtest_output_file}` when you need to drill down further. There are 6 sub-commands of `uv run python -m mbt.run.review ${backtest_output_file}`:
  - `summary`: `python -m mbt.run.review ${backtest_output_file} summary [-l 10]` to output the Markdown statistical summary and Top/Bottom objs table.
  - `trades`: `python -m mbt.run.review ${backtest_output_file} trades [-s ${obj}] [-l 10]` to review portfolio trade cycles (holding periods, entry/exit dates, profit rate, and profit amount), optionally filtered by obj.
  - `attribution`: `python -m mbt.run.review ${backtest_output_file} attribution [-o desc|asc] [-l 10]` to review per-stock contribution ranking (trade count, win rate, total profit amount, capital contribution rate, and avg return), sorted by total profit.
  - `stat`: `python -m mbt.run.review ${backtest_output_file} stat` to review the statistics of the metrics in JSON format.
  - `detail`: `python -m mbt.run.review ${backtest_output_file} detail -s ${obj}` to review the detailed bar-by-bar series and metrics of a specific obj.
  - `sort`: `python -m mbt.run.review ${backtest_output_file} sort -m ${metric} [-o desc|asc] [-l 10]` to show a specific metric sorted. the metric can be any of:
    - `profit` : the final profit of the strategy.
    - `profit_original` : the final profit during the test but without apply the strategy.
    - `benchmark_profit` : the final profit of the benchmark index over the same period.
    - `alpha` : the alpha of the strategy.
    - `beta` : the beta of the strategy.
    - `sharpe` : the sharpe ratio of the strategy.
    - `information_ratio` : the information ratio of the strategy.
    - `ir` : the information ratio of the strategy.
    - `sortino` : the sortino ratio of the strategy.
    - `treynor` : the treynor ratio of the strategy.
    - `max_drawdown` : the maximum drawdown of the strategy.
    - `calmar` : the calmar ratio of the strategy.
    - `win_rate` : the win rate of the strategy (trade-level in portfolio mode).
    - `trade_win_rate` : the win rate across trade cycles / holding periods.
    - `daily_win_rate` : the bar-level return win rate.
    - `avg_win` : the average win of the strategy.
    - `avg_loss` : the average loss of the strategy.
    - `profit_factor` : the profit factor of the strategy.
    - `fee_rate` : the fee rate of the strategy.
    - `trade_count` : the trade count of the strategy.
    - `rebalance_count` : the rebalance bar count of the strategy.
- All review result are JSON format and printed to stdout.
- You may run the review commands multiple times to explore the data.
- Write a report to analyze the metrics to `report.${round_id}.md` using chinese, mention the key findings and insights.
- Point out the shortcomings and suggests to improve.
- Use the report, rewrite a strategy file for next round

## Full Report

- Use chinese when writing the full report, write it to `report.md`.
- The report should include the following sections:
  - Strategy Introduction
  - Backtest results summary, with top 10 better and worse objs.
  - Optimization process and key insights
  - Explanation of advantages
  - Reflection on the shortcomings and limitations
  - Conclusion and outlook
