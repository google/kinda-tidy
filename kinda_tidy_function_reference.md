# Kinda-tidy Function Reference

<!--*
# Document freshness: For more information, see go/fresh-source.
freshness: { owner: 'agkeck' reviewed: '2022-10-17' }
*-->

go/kinda-tidy-functions

[TOC]

## Filter/Select

### **select_columns(columns, flags=None)**

Select a subset of columns based on one or more regular expressions matching
column names.

columns
:   A string or list of strings. The strings are used as regular expressions and
    any column whose name matches a supplied regular expression is in the
    returned dataframe. Note that the match is `matchall` so the expression must
    match the entire name.

flags
:   Optional [flags](https://docs.python.org/3/library/re.html#flags) can be
    sent to the regular expression compiler.

--------------------------------------------------------------------------------

### **select_rows(predicate)**

Select a subset of rows by applying a function returning a vector of booleans to
the dataframe.

predicate
:   A function which takes the current dataframe and returns a vector of
    booleans used as a mask to select rows of the dataframe

--------------------------------------------------------------------------------

### **query(query_string)**

Select a subset of columns by evaluating a boolean expression for each row.
[built in method](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html)

query_string
:   A string describing a boolean condition to be evaluated on each row. The
    execution environment for the evaluation is not that of the calling scope so
    this is limited to fairly simple conditions.

## Mutation

### **assign(kwargs)**

Add one or more new columns using a function to compute the new column values
from the existing column values.
[built in method](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.assign.html)
extended by kinda-tidy to assign in groups.

kwargs
:   one or more assignments to new columns in the returned dataframe.
    Assignments are of the form `column_name = lambda df: code` where `code`
    creates the new column from the columns of the input dataframe. Kinda-tidy
    extend this functionality to grouped dataframes so one can make reference to
    aggregate functions on groups (i.e. max value in a group) If the new column
    name is the same as an existing column the new values replace the old.

## Group

### **tidy_groupby(columns)**

Thinly wrapped version of
[built in method](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html)

columns
:   name or list of names of columns to group the dataframe by for subsequent
    operations. `tidy_groupby()` wraps the built in function to suppress the
    pandas behavior of constructing indices from the grouping columns. Since
    these indices are generally dropped immediately this is a convenience
    function.

## Sample

### **equisample(grouping, n=None, kwargs)**

Draw a random sample from the dataframe sampling equally from the levels
specified by the `grouping` parameter.

grouping
:   name or list of names of columns specifying the levels for equal sampling.

n
:   Optional, if given it specifies the size of the sample to be drawn from each
    level. If missing the largest sample that can be taken equally from all
    levels is made.

kwargs
:   additional keyword arguments passed to pandas `sample()`
    [built in method](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html).

## Restructure

### **set_categorical(category_col, value_col)**

Converts a column to categorical type with levels order by values of a second
column. This can be used with a grouped dataframe (from `tidy_groupby()`) to
order levels within groups and to select top_n levels. In these cases we attempt
to keep overall order of grouping with levels of the target column ordered
within the overall ordering.

category_col
:   column to be made categorical type.

value_col
:   column used to order the levels. The dataframe is grouped by `category_col`
    and `value_col` is aggregated with an aggregation function. The levels are
    ordered by this aggregation descending. This is typically used to convert a
    column of strings to a categorical with levels ordered by decreasing
    importance. The order is then used in visualizations to carry additional
    information. If `value_col` is missing the levels are ordered alphabetically
    by label.

*optional arguments*:

aggregator = np.sum
:   A function to be applied to groups to determine the ordering of the levels
    of the categorical.

reverse = False
:   reverse the order of levels

top_n = None
:   An integer giving the number of levels to retain. Other levels are coalesced
    to a level with label `other_label`. top_n is calculated based on the
    aggregation so if one wants a 'bottom n' one should negate the aggregation
    function.

keep_other = False
:   Should the other category be retained or discarded.

other_label = '_other'
:   label for the other category.

### **reverse_categories(target_columns)**

Reverse the order of levels for one or more categorical columns

target_columns
:   the name or list of names of the columns to have their levels reversed. If
    missing, all categorical columns are reversed.

### **drop_unused_levels(target_columns)**

Remove unused levels from one or more categorical columns.

target_columns
:   the name or list of names of the columns to have their unused levels
    removed. If missing, all categorical columns are reversed.

### **flatten_columns()**

Convert a multi index column index to a one dimensional list of column names by
concatenating the levels of the multi-index. This is useful when aggregating
columns with multiple aggregation functions e.g. `.agg({'col':[np.mean,
np.median]})`

*optional arguments*:

sep = '_'
:   The string to use between elements in the concatenation of levels.

### **melt(id_vars, value_vars, var_name, value_name)**

Convert wide to tall format. Collects one group of columns into two columns:
labels, and values.
[built in method](https://pandas.pydata.org/docs/reference/api/pandas.melt.html)

id_vars
:   variables in current dataframe to be repeated as necessary as the columns
    are stacked.

value_vars
:   The columns to be stacked up into two columns: labels and values

var_name
:   The name to use for the new column of labels. The labels come from the
    column names in `value_vars`

value_name
:   The name to use for the new column of values. The values come from the
    content of `value_vars`.

### **pivot(index, columns, values)**

Convert tall to wide format.
[built in method](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot.html)

index
:   columns to form the index of the new dataframe

columns
:   name or list of names of columns whose contents will generate names for new
    columns.

values
:   name or names of columns whose contents will provide the values of the new
    columns.

### **tee(function, print_result=True)**

Pass the dataframe through while applying a function and printing the results.

function
:   function to apply to the dataframe, the results of which may be printed

print_results
:   should the result of the function be printed.

## Join/Merge

### **merge(other_table)**

Join current dataframe with another table. The pandas method `join()` uses
indices and is not useful for tidy analysis, use `merge()` instead.
[built in method](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html)

other_table
:   dataframe with the right hand side of the join

*optional arguments*:

how = 'inner'
:   type of join \{‘left’, ‘right’, ‘outer’, ‘inner’, ‘cross’\}

on
:   a name or list of names of columns shared between the two tables to join on.

left_on, right_on
:   as with on when the names differ between the tables

## Plot with ggplot

## Plot with Altair

### **alt_chart()**

Wrapper function that allows to use Altair Chart as part of DataFrame pipe.

Example usage:

```
(df
 .alt_chart()
 .mark_line()
 .encode(
    x='date',
    y='wt'
    )
)
```

### **alt_horizontal_line()**

"Utility to plot a horizontal line in Altair on top of existing chart.

*optional arguments*:

y_value = 0.0
:   Value where the horizontal line intersects y-axis.

y = 'y'
:   Name of the column (default should be fine for most cases).

color = 'black'
:   Color of the line, can be a string accepted by Vega lite or HEX.

size = 2
:   Width of the line.

line_kwargs
:   Additional keyword arguments passed to mark_rule().

E.g. usage:

```
(df
 .alt_chart()
 .mark_line()
 .encode(
    x='date',
    y='wt'
    )
 + alt_horizontal_line()
)
```

### **alt_vertical_line()**

Utility to plot a vertical line in Altair on top of existing chart. Part of
`alt_utils.py` package.

*optional arguments*:

x_value = 0.0
:   Value where the horizontal line intersects x-axis.

x = 'x'
:   Name of the column (default should be fine for most cases).

color = 'black'
:   Color of the line, can be a string accepted by Vega lite or HEX.

size = 2
:   Width of the line.

line_kwargs
:   Additional keyword arguments passed to mark_rule().

E.g. usage:

```
(df
 .alt_chart()
 .mark_line()
 .encode(
    x='date',
    y='wt'
    )
 + alt_vertical_line()
)
```

### **alt_diagonal_line()**

Utility to plot a diagonal line in Altair on top of existing chart. Part of
`alt_utils.py` package.

*optional arguments*:

x_start_end_list = (0, 1)
:   start and end values on x axis; can technically draw any arbitrary line by
    specifying corresponding x and y values.

y_start_end_list = (0, 1)
:   start and end values on y axis

x = 'x'
:   Name of the x column (default should be fine for most cases).

y = 'y'
:   Name of the y column (default should be fine for most cases).

color = 'black'
:   Color of the line, can be a string accepted by Vega lite or HEX.

size = 2
:   Width of the line.

opacity = 0.4,
:   Line opacity.

opacity = [5, 5]
:   Stroke type of the line, default is `- - - -`.

line_kwargs
:   Additional keyword arguments passed to mark_rule().

E.g. usage: add a diagonal baseline dashed line to ROC curve:

```
(df
 .alt_chart()
 .mark_line()
 .encode(
    x='false_positive_rate',
    y='true_positive_rate'
    )
 + alt_diagonal_line()
)
```
