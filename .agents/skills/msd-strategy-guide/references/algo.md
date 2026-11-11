
The input and output arrays contain multiple stocks/objects. The algorithm implementations compute in parallel to improve performance, so you should use these algorithms as much as possible instead of implementing them yourself.

These arrays are organized as follows: let A and B be two stocks, respectively, and let close be the data. Then close = [A0, A1, ..., An, B0, B1, ..., Bn].
When you need to write your own algorithm, please make sure to preserve this structure.

The algorithm library is an open-source project. The installation package is named `py-alpha-lib`, and after installation, the import package name is `alpha`. You can use Python's Doc mechanism to view its functions list. eg. `python -m pydoc alpha`

Its repository addresses are:

- [Github](https://github.com/msd-rs/py-alpha-lib/tree/main/py-binding)
  - [Curated algorithm list](https://raw.githubusercontent.com/msd-rs/py-alpha-lib/refs/heads/main/py-binding/python/alpha/algo.md)
  - For each algorithm, if you need to further explore its behavior, you can read its integration test code. For example, for algorithm MA, see the [integration test code](https://raw.githubusercontent.com/msd-rs/py-alpha-lib/refs/heads/main/py-binding/itests/test_ma.py), which includes a brief introduction to the algorithm, examples of different scenarios, etc. You can build the URL of integration test code like above by replace the algorithm name.
- [CNB](https://cnb.cool/elsejj/msd/alpha-lib/-/tree/main/py-binding): China mirror
  - [Curated algorithm list](https://cnb.cool/elsejj/msd/alpha-lib/-/git/raw/main/py-binding/python/alpha/algo.md)
  - For each algorithm, if you need to further explore its behavior, you can read its integration test code. For example, for algorithm MA, see the [integration test code](https://cnb.cool/elsejj/msd/alpha-lib/-/git/raw/main/py-binding/itests/test_ma.py), which includes a brief introduction to the algorithm, examples of different scenarios, etc. You can build the URL of integration test code like above by replace the algorithm name.

# Context Handling

Because `alpha` compute is paralleled, the context manager is very important, it control the algorithm's behavior.

- groups: how many stocks, each group will be compute as a no cross chunk.
- start/end: for a stock chunk, the internal range to compute, default full range
- flags: 
  - FLAG_SKIP_NAN: make NaN value take no effect when compute, it's usefully to handle joined data
  - FLAG_STRICTLY_CYCLE: no fallback when window on data is not enough length, eg, MA(3) on a series, without fallback the first 2 result will be NaN because window is not enough, with fallback it will have value with current window length.

To Use `alpha.Context`

- Short lifecycle: Use `with ctx:` for automatic management and contamination prevention.

```python
import alpha as al
ctx = al.Context(groups=100, flags=al.FLAG_SKIP_NAN)
with ctx:
  res = al.MA(data, 3)
# context will be restore after exit with
```


- Long lifecycle: persisted within a process or class and activated on demand.

```python
class FactorPipeline:
  def __init__(self, num_securities: int):
    # keep a Context 
    self.ctx = al.Context(groups=num_securities)
  def compute_step1(self, x):
    with self.ctx:
      return al.MA(x, 3)
  def compute_step2(self, y):
    with self.ctx:
      return al.EMA(y, 10)
```

- Nested scope (local shadowing)

```python
with al.Context(groups=100):
    r1 = al.CC_RANK(data1)  # groups=100
    with al.Context(groups=10):
        r2 = al.CC_RANK(data2)  # local shadow groups=10
    r3 = al.CC_RANK(data3)  # auto restore to groups=100
```