Extends pandas DataFrames for common functionality mainly around MultiIndex.

* access columns matching on any level
* access columns using regex
* distribute calculations for repeating sub-levels 

But also some missing methods get extended like:

* cumpct_change
* cumapply
* rolling apply for multiple return values optionally parallel 

For more use cases check the notebooks in the [examples][gh1] directory.

[gh1]: https://github.com/Pandas-Quant-Finance/pandas-df-commons/tree/master/examples/

```python
from pandas_df_commons.patched import pd
import numpy as np

df = pd.concat(
    [pd.DataFrame({"Samples1": np.random.normal(0, 1, 100), "Samples2": np.random.normal(0, 1, 100)})],
    axis=1,
    keys=["A", "B"]
)

# Easy access to the lower levels
df.X["Samples1"]

# Also Regex between ~/:regex:/ are suported
df.X["~/.*2$/"]
```