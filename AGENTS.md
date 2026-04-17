This project is a stock strategy backtest framework. 

# Technologies

- Use `uv` for dependency management and environment management.
- The main module is `mbt` in the root directory.
- Use 'pytest' for testing, the source code for tests is in 'tests' directory.


# Project Structure

- `mbt`: source code
  - `account.py`: simulates account holding and cash.
  - `data_provider.py`: base class of data provider to retrieve multiple data series over time for multiple symbols.
  - `strategy.py`: base class of strategy.
  - `run/`: a strategy runner based on `msd` database and `python-alpha-lib` algorithms library.
    - `cli.py`: a command line interface to run a strategy.
    - `server.py`: a http server to run a strategy.
    - `msd.py`: a module to load data from `msd` database.
- `tests/`: test code for `mbt`.
- `strategy`: some example strategies.

# How to Run

## CLI

use `uv python -m mbt.run.cli` to run a strategy from command line.
use `uv python -m mbt.run.cli -h` to see the available options and arguments.

## Server

use `uv python -m mbt.run.server` to start a http server.
use `uv python -m mbt.run.server -h` to see the available options and arguments.


# Dev Environment

- The `msd` database server address is `http://localhost:50510`.
- The stock symbol is Chinese stock symbol, e.g. `SH600000`, `SZ000001`.