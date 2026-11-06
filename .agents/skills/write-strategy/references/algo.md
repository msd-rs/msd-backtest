
The input and output arrays contain multiple stocks/objects (specified via `aplha.set_ctx(groups)`). The algorithm implementations compute in parallel to improve performance, so you should use these algorithms as much as possible instead of implementing them yourself.

These arrays are organized as follows: let A and B be two stocks, respectively, and let close be the data. Then close = [A0, A1, ..., An, B0, B1, ..., Bn].
When you need to write your own algorithm, please make sure to preserve this structure.

The algorithm library is an open-source project. The installation package is named `py-alpha-lib`, and after installation, the import package name is `alpha`. You can use Python's Doc mechanism to view its functions list.

Its repository addresses are:

- [Github](https://github.com/msd-rs/py-alpha-lib/tree/main/py-binding)
  - [Curated algorithm list](https://raw.githubusercontent.com/msd-rs/py-alpha-lib/refs/heads/main/py-binding/python/alpha/algo.md)
  - For each algorithm, if you need to further explore its behavior, you can read its integration test code. For example, for algorithm MA, see the [integration test code](https://raw.githubusercontent.com/msd-rs/py-alpha-lib/refs/heads/main/py-binding/itests/test_ma.py), which includes a brief introduction to the algorithm, examples of different scenarios, etc.
- [CNB](https://cnb.cool/elsejj/msd/alpha-lib/-/tree/main/py-binding): China mirror
  - [Curated algorithm list](https://cnb.cool/elsejj/msd/alpha-lib/-/git/raw/main/py-binding/python/alpha/algo.md)
  - For each algorithm, if you need to further explore its behavior, you can read its integration test code. For example, for algorithm MA, see the [integration test code](https://cnb.cool/elsejj/msd/alpha-lib/-/git/raw/main/py-binding/itests/test_ma.py), which includes a brief introduction to the algorithm, examples of different scenarios, etc.



