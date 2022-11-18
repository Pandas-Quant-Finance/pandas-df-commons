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