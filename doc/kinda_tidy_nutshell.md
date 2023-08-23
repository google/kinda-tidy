# Kinda-tidy Tidyverse style data analysis in python

<!--*
# Document freshness: For more information, see go/fresh-source.
freshness: { owner: 'agkeck' reviewed: '2022-10-17' }
*-->

[TOC]

## Introducing the tidyverse

### Principles

The tidyverse consists of a set of functions and a pipelining system that allows
one to write data manipulation code focussing on the operations (verbs) with
minimal attention to the containers (nouns). We obtain this focus on the
operations by enforcing two principles:

*   Values are *immutable*. Once a value has been assigned to an identifier that
    identifier will always contain the same value, and

*   Functions are *pure*. A function's result depends only on the arguments
    supplied and neither depends on nor changes any other states of the program.
    There can be no side-effects of an operation to keep track of.

<!-- **NOTE**: How does R do it? R has two critical features: the ability to define
new binary operations such as a pipe `%>%`, and lazy evaluation that protects
references to columns within a dataframe from evaluation in outer scope of
function calls.

**NOTE**: How do we do it in python? We add methods to the DataFrame class so
that we can get pipelines by chaining method calls. We protect arguments with
either anonymous functions (`lambda df: ...`) or quotes.
 -->

There are two consequences of immutable values. First one always knows what is
held in an identifier---they need not consider the execution history when using
it. Second, since one can't change what is in an identifier they will be more
careful about defining it in the first place. Programs built in this style name
very few intermediate values and hence tend to consist of a single noun and a
number of verbs.

Pure functions make code easy to reason about. One does not have to consider
anything but what is directly present in the argument list to understand the
operation. Moreover, when applied to pipeline segments purity implies that these
segments can be reused in other pipelines with confidence. In fact this gives
rise to one of the greatest efficiencies of the tidy verse. Consider an analysis
with a log transformed response. If one wants to repeat the analysis with a
square root transform one can duplicate the pipeline, replace the one segment of
interest and execute secure in the knowledge that there is no interaction
between the two analyses.

IMPORTANT: Many people worry that immutable values and pure functions will lose
the efficiency of working *in place*. **Don't worry about efficiency!**

IMPORTANT: Still, a little reuse here and there can't hurt and will improve
efficiency. **Don't worry about efficiency!** and review
[Knuth](https://wiki.c2.com/?PrematureOptimization).

IMPORTANT: But the environment? Don't I owe it to the world to modify data
frames in place to reduce resource usage? **Don't worry about efficiency!**
Pandas ignores the `inplace=True` suggestion in many cases. Furthermore a well
designed language like Haskell or Julia has the smarts to `inplace` as
appropriate without suggestions so if we really care about the environment we
should not write in python.

### Getting Started

Most of the functionality we need is built into panda dataframes. The remaining
methods are added by the `kinda-tidy` package which can be loaded as below.
```python
# install the package from github
!pip install git+https://github.com/google/kinda-tidy.git
```

```python
import pandas as pd
import numpy as np

# for ggplot
from plotnine import *
from mizani.breaks import date_breaks
from mizani.formatters import date_format, percent_format, currency_format

# for kinda-tidy
import kinda_tidy
```

For the examples in this document we'll use data from the gapminder project, a
collection of socio-economic indicators for 142 countries.

```python
gapminder = pd.read_csv('https://github.com/google/kinda-tidy/blob/main/data/gapminder.csv?raw=true')
gapminder.sample(5)
```

giving the following dataframe:

country     | continent | year | lifeexp | population | gdppercap
----------- | --------- | ---- | ------- | -------- | ---------
Afghanistan | Asia      | 1952 | 28.801  | 8425333  | 779.445
Afghanistan | Asia      | 1957 | 30.332  | 9240934  | 820.853
Afghanistan | Asia      | 1962 | 31.997  | 10267083 | 853.101
Afghanistan | Asia      | 1967 | 34.02   | 11537966 | 836.197
Afghanistan | Asia      | 1972 | 36.088  | 13079460 | 739.981

We'll also need a wide dataframe so we pivot the population column against year

```python {.ignore-codeblockanalysis}
gapminder_pop = (gapminder
  .pivot(index='country', columns='year', values='population')
  .reset_index()
  .rename(columns=lambda cname: 'year:' + str(cname))
  .rename(columns={'year:country': 'country'})  # fix first column name
)
gapminder_pop
```

giving

country     | year:1952 | year:1957 | year:1962 | year:1967 | year:1972 | year:1977 | year:1982 | year:1987 | year:1992 | year:1997 | year:2002 | year:2007
----------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | ---------
Afghanistan | 8425333   | 9240934   | 10267083  | 11537966  | 13079460  | 14880372  | 12881816  | 13867957  | 16317921  | 22227415  | 25268405  | 31889923
Albania     | 1282697   | 1476505   | 1728137   | 1984060   | 2263554   | 2509048   | 2780097   | 3075321   | 3326498   | 3428038   | 3508512   | 3600523
Algeria     | 9279525   | 10270856  | 11000948  | 12760499  | 14760787  | 17152804  | 20033753  | 23254956  | 26298373  | 29072015  | 31287142  | 33333216
Angola      | 4232095   | 4561361   | 4826015   | 5247469   | 5894858   | 6162675   | 7016384   | 7874230   | 8735988   | 9875024   | 10866106  | 12420476
Argentina   | 17876956  | 19610538  | 21283783  | 22934225  | 24779799  | 26983828  | 29341374  | 31620918  | 33958947  | 36203463  | 38331121  | 40301927

## Filter/Select

Pandas has facilities for filtering dataframes by indices if you set your
dataframe with appropriate indices. In general, however, the most common tasks
are filtering columns by name and rows by content so we offer those use cases
below.

### Columns

Pandas allows one to select or drop columns by name. We expose the
`select_columns()` method to select columns by name using a singleton or list of
literal names or regular expressions. This method returns a copy of the
dataframe to avoid risk of the
[chained assignment problem](https://pandas.pydata.org/pandas-docs/version/0.22/indexing.html#indexing-view-versus-copy).

```python
gapminder.select_columns('country')
# select two columns
gapminder.select_columns(['country', 'year'])
# from the wide dataframe select all population columns
gapminder_pop.select_columns('year:.*')
# country and population from the 1950s and 1960s
gapminder_pop.select_columns(['country', '.*19(5|6)\d'])
```

TIP: The regular expression is matched again the full column name so *starts
with year* is expressed as `'year.*'` and *ends with 2* is expressed as `'.*2'`.
Also note that the order of result columns is that of the underlying dataframe.

### Rows

Pandas built in `query()`
[method](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html)
is quite flexible and most conditions can be expressed easily:

```python
gapminder.query('year >= 1970')
gapminder.query('year >= 1970 and continent == "Asia"')
```

While `query()` handles the bulk of tasks, there are times when the execution
environment where the condition is evaluated creates problems. Consider, *e.g.*
a comparison on log scale:

```python
# this fails as np is not in evaluation scope even though it is in outer scope
gapminder.query('np.log(year) > 7.5857888 and continent == "Asia"')
```

For cases where we want complete control over creating the boolean mask for
selection we expose a `select_rows()` method. `select_rows()` takes an anonymous
function with signature `pd.DataFrame -> Sequence[bool]`, applies the function
to the DataFrame, and returns the rows corresponding to `True` in the result
vector. Thus we can select the appropriate rows with

```python
gapminder.select_rows(lambda df: np.log(df.year) >= 7.5857888)
```

With the `query()` method the selection expression is evaluated for each row but
with `select_rows()` the function occurs on columns. One major downside of this
is that boolean operators are not automatically threaded over pandas series
objects. This means that to find Asian countries since 1970 we would invoke the
`np.logical_and()` to thread the conjunction:

```python
gapminder.select_rows(lambda df: np.logical_and(
    np.log(df.year) >= 7.5857888, df.continent == 'Asia'))
```

## Mutation

Pandas provides the `assign()` method for creating/replacing columns based on
dataframe values. We extend this functionality to grouped dataframes so that we
can apply arbitrary functions of groups to produce new values. Some examples:

--------------------------------------------------------------------------------

Example:

```python
(gapminder
  .assign(gdp=lambda df: df.gdppercap * df.population)
)
```

creates a `gdp` column from the `gdppercap` and `population` columns.

--------------------------------------------------------------------------------

Example:

```python
(gapminder
  .assign(population=lambda df: df.population / 1e6)
 # modeling and visualization here
)
```

Rescales population to be measured in millions. Recall the principle that values
are immutable, here we seem to have mutated the gapminder dataframe. But the
code creates a copy of the dataframe so we have created a new, anonymous
dataframe that has a rescaled population. This temporary dataframe will exist
only in the scope of the parens. We will perform our modelling and visualization
within these parens so the creation and consumption in the model all occur in a
single block that can be understood with reference only to the immutable
`gapminder` dataframe and the various verbs applied.

--------------------------------------------------------------------------------

Example:

```python {.ignore-codeblockanalysis}
(gapminder
  .tidy_groupby(['continent','year'])
  .assign(continent_pop = lambda df: sum(df.population))
  .sort_values('year') # ensure order for pulling first and last values
  .tidy_groupby(['continent'])
  .assign(growth_factor = lambda df:df.continent_pop.iloc[-1]/df.continent_pop.iloc[0],
          time_span = lambda df:df.year.iloc[-1] - df.year.iloc[0])
 # create annualized growth rate from final = initial (1+r)^time_span
 .assign(continent_pop_annualized_growth=lambda df: df.growth_factor**
         (1 / df.time_span) - 1))
```
creates an annualized population growth rate for each continent. This involves
creating `<continent, year>` total population, sorting to ensure that the data
are in year order (grouping will respect the ordering), regrouping by continent
to get `<continent>` level annualized growth rate.

## Group

Pandas allows grouped operations on dataframes. Unfortunately pandas
automatically adds an index to the results requiring one to `reset_index()`
after every grouped operation. We introduce `tidy_groupby()` as a replacement
that does not create the unnecessary indices. As a wrapper around `groupby()` it
supports the same arguments. The grouping can be made on any collection of
columns. By default pandas supports

*   aggregation using scalar valued functions to a new dataframe with one row
    for each group
*   application of a function to each group, the results of these function
    applications are gathered into a series or dataframe if possible. Most of
    these uses are subsumed by the kinda-tidy extensions

Kinda-tidy adds two more operations

*   group assignment using group level aggregate functions e.g. normalizing a
    column making reference to the group mean and standard deviation.
*   group row selection using group level aggregate functions to specify the
    rows of interest e.g. getting those rows in the top 90th percentile.

--------------------------------------------------------------------------------

Example:

```python
(gapminder
  .tidy_groupby('country').agg({'population': np.mean})
)
```
aggregates the population column to a country level mean. We reset the index
because groupby creates an index and it is rarely if ever useful afterward.

--------------------------------------------------------------------------------

Example:

```python
(gapminder
  .tidy_groupby('country')
  .agg(mean_pop=('population', np.mean), row_count=('population', 'count'))
 )
```

similarly aggregates. In this case we can create new names for the columns and
also give two different aggregations in a convenient way.

--------------------------------------------------------------------------------

Example:

```python {.ignore-codeblockanalysis}
(gapminder
  .tidy_groupby('country')
  .agg({'population':[np.mean, np.median]})
  .flatten_columns()
 )
```

uses a list of aggregations for the single column. This results in a two level
index on the columns. This structure is rarely useful so kinda-tidy provides a
method `flatten_columns()` that flattens the 2 level index into ordinary column
names.

Example:

```python {.ignore-codeblockanalysis}
(gapminder
  .sort_values('year')
  .tidy_groupby('country')
  .assign(normalized_pop = lambda df: df.population/df.population.iloc[0])
 )
```

creates a population column normalized to the first available population (1952).

## Restructure

### Categorical data

### Tall/Wide transformations


## Join/Merge

The pandas `join()` method works nicely with indices.  As mentioned before, indices are powerful but non intuitive to those coming from the tidyverse.  The method `merge()` performs the same task but it is more like SQL and dplyr's `join()` method.

---
The basic syntax is

```
left_dataframe.merge(right_dataframe, on= , how=)
```
Where `on` specifies the key columns for the join and `how` specifies `inner`, `outer`, `left` or `right`.

In this case we wish to augment the dataframe with yearly continent wide population.  We first create the yearly continent totals and then join with the original on continent and year.  We rename the column name to avoid name collison in the join although pandas will add suffixes to ensure unique names.

While we are at it we can add a new column with the fraction of the continent's total population represented by each country each year.

```python
(gapminder
  .groupby(['continent', 'year'])
  .agg({'pop':np.sum})
  .reset_index()
  .rename(columns={'pop':'continental_pop'})
  .merge(gapminder, on=['continent', 'year'], how='right')
  .assign(pop_frac = lambda df: df['pop']/df.continental_pop)
)
```
The above example uses a join to perform what is essentially an analytic function.  A more straightforward approach would be to apply a function that assigns group totals to a grouped object as follows.

```python
  (gapminder
   .groupby(['continent', 'year'])
   .apply(lambda grp: grp.assign(continental_pop = np.sum(grp['pop'])))
   .assign(pop_frac = lambda df: df['pop']/df['continental_pop'])
   .reset_index(drop=True)
  )
```

## Plot with ggplot

## Plot with Altair

## Extending kinda-tidy

## Why is most pandas code so bad? (A rant)

Suppose we have a dataframe with `watchtime` in ms and we want it in minutes.
The standard pandas way to express this is

```python
saturday_am_childrens_watch[
    'watchtime'] = saturday_am_childrens_watch['watchtime'] / (1000 * 60)
```

How do I loathe this? Let me count the ways:

1.  It takes ~100 characters to express an operation of 10 characters
    `/(1000*60)` and a target of 9 characters `watchtime`.
1.  It can only be used with a dataframe of the name
    `saturday_am_childrens_watch`. Granted, our coding suggestions are getting
    much better so we will soon be just pressing tab to get the boilerplate for
    `sunday_am_childrens_watch` but we must ask ourselves *Why are we teaching
    our coding assistants this?* or perhaps more to the point *What must our
    models think of us for programming like this?* We should really clean the
    house before the housekeepers get here because we don't want them to know
    that this is how we live.
1.  We have just doubled the number of entities we must keep track of.
    [William of Ockham](https://en.wikipedia.org/wiki/Occam%27s_razor) warned us
    *Entities are not to be multiplied without necessity.* Before there was just
    `saturday_am_childrens_watch` now there is `saturday_am_childrens_watch
    before conversion` and `saturday_am_childrens_watch after conversion`. Every
    time we look at `saturday_am_childrens_watch` we have to ask ourselves which
    version is it? If we are working in colab we can't even answer that by
    looking at the code since we could have evaluated the cells out of top to
    bottom order. So we are forced to distinguish between two indistinguishable
    symbols based on our own memory of our actions in the past.
1.  The operation is not idempotent. If we accidentally execute the line twice
    our column is now in kilo-hours. The alternative is to give new names to
    every derived column. Aside from the problem of creating new and
    increasingly lengthy names for derived information we now have redundant
    forms of the same underlying data.
