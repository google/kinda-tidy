# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tidyverse like operations for Python."""


import numbers
import re
from typing import Any, Callable, Optional, Sequence, Union

import altair as alt
import numpy as np
import pandas as pd
import plotnine as gg
import statsmodels.api as sm


Number = numbers.Number


def _adapt_scalar_to_vector(value: Any) -> Sequence[Any]:
  """Adapts a scalar input to a one element list if needed."""
  if np.isscalar(value):
    return [value]
  else:
    return value


def mean_weighted(values: Sequence[Number],
                  weights: Sequence[Number]) -> Number:
  if any(w < 0 for w in weights):
    raise ValueError('Must pass non-negative weights.')
  else:
    return sm.stats.DescrStatsW(values, weights).mean


def _set_categorical(self: pd.DataFrame,
                     category_col: str,
                     value_col: Optional[str] = None,
                     aggregator: Callable[[pd.Series], Number] = np.sum,
                     reverse: bool = False,
                     top_n: Optional[int] = None,
                     keep_other: bool = False,
                     other_label: str = '_other') -> pd.DataFrame:
  """Converts a column to categorical type.

  Args:
    self: pandas DataFrame
    category_col: name of column to be converted to category
    value_col: name of column with numeric entries used to order the levels of
      the category column.  By default the sum of this column on each category
      is used to arrange the levels (in descending order). Note that this does
      not make an ordered categorical column, it assigns levels in an order that
      is respected by ggplot functions. value_col is not strictly required and
      if missing the levels are ordered alphabetically.
    aggregator: a function to apply to the value_col for each level of
      category_col determining the ordering.
    reverse: should the levels be reversed to show smallest (of top_n) first?
    top_n: Optional positive integers with the number of levels to retain.  All
      other data are assigned level 'other'.
    keep_other: Should the 'other' category be retained?
    other_label: label to use for the 'other' category.

  Returns:
    A DataFrame with category_col converted to a categorical series. The order
    of the categories is given by applying the function aggregator to the column
    value_col after grouping by category_col. One can reduce the number of
    categories by keeeping only the top_n and optionally labelling or dropping
    the others. The order of the categories can be reversed with the reverse
    keyword.
  """

  def sort_for_categories(df):
    # pyformat: disable
    return (df.groupby(category_col)
            .agg({value_col: aggregator})
            .reset_index()
            .sort_values(value_col, ascending=False))
    # pyformat: enable

  res = self.copy()

  # recast to string if the column is currently categorical
  if pd.api.types.is_categorical_dtype(res[category_col]):
    res[category_col] = res[category_col].astype(str)

  if not value_col:
    res.loc[:, category_col] = pd.Categorical(res[category_col])
    return res

  sort_data = sort_for_categories(res)

  categories = sort_data[category_col][:top_n]  # note: x[:None] = x
  # relabel or remove others
  if top_n:
    keep_row_mask = res[category_col].isin(categories)
    if keep_other:
      res.loc[~keep_row_mask, category_col] = other_label
    else:
      res = res[keep_row_mask].copy()
  # recalculate order with possible new _other category
  final_categories = sort_for_categories(res)[category_col]
  if reverse:
    final_categories = final_categories[::-1]

  res.loc[:, category_col] = pd.Categorical(
      res[category_col], categories=final_categories)

  return res


def _groupby_set_categorical(self: pd.DataFrame,
                             category_col: str,
                             value_col: Optional[str] = None,
                             aggregator: Callable[[pd.Series], Number] = np.sum,
                             reverse: bool = False,
                             top_n: Optional[int] = None,
                             keep_other: bool = False,
                             other_label: str = '_other') -> pd.DataFrame:
  """Converts a column to categorical type in a grouped data frame.

  Args:
    self: pandas DataFrame
    category_col: name of column to be converted to category
    value_col: name of column with numeric entries used to order the levels of
      the category column.  By default the sum of this column on each category
      is used to arrange the levels (in descending order). Note that this does
      not make an ordered categorical column, it assigns levels in an order that
      is respected by ggplot functions. value_col is not strictly required and
      if missing the levels are ordered alphabetically.
    aggregator: a function to apply to the value_col for each level of
      category_col determining the ordering.
    reverse: should the levels be reversed to show smallest (of top_n) first?
    top_n: Optional positive integers with the number of levels to retain.  All
      other data are assigned level 'other'.
    keep_other: Should the 'other' category be retained?
    other_label: label to use for the 'other' category.

  Returns:
    A DataFrame with category_col converted to a categorical series. This works
    on grouped data frames so we take some care to ensure that the ordering of
    levels is grouped within the groupby levels.  For hierarchical data where
    the category column is nested within groups of the grouped object they are
    ordered by the aggregator acting on the value column and these groupings
    are ordered by the aggregator acting on the value column of the groupby
    groups.  For example if the data frame is grouped by continent and we want
    to make country name categorical based on population we would get country
    levels as asia_country_rank_1, asia_country_rank_2, ...
    americas_country_rank_1, americas_country_rank_2, ... since the total
    population of continents is ordered Asia, Americas, ... .
    For non-hierarchical data where a category is present in more than one
    level of the grouping, it will be sorted with the largest group to which
    it belongs.
  """
  grouping_columns = self.grouper.names

  def act_on_groups(grp):
    return grp.set_categorical(category_col, value_col, aggregator, reverse,
                               top_n, keep_other, other_label)

  result = self.apply(act_on_groups)

  # create order for levels
  # generate groupby level aggregations and groupby + category_col aggregation
  # to order categorical labels by (groupby aggregation, category aggregation)
  # pyformat: disable
  order_df = (
      result
      .tidy_groupby(grouping_columns)
      .assign(_group_aggregation=lambda grp: aggregator(grp[value_col]))
      # the observed=true  protects us if the grouping columns
      # contain categoricals.  In such cases the groupby creates a cross product
      # of levels even if there is no data is some levels.
      .tidy_groupby(grouping_columns + [category_col], observed=True)
      .agg({'_group_aggregation': 'first', value_col: aggregator})
      .sort_values(['_group_aggregation', value_col], ascending=False)
      )
  # pyformat: enable
  levels = pd.unique(order_df[category_col])
  if reverse:
    levels = levels[::-1]

  result[category_col] = pd.Categorical(result[category_col], categories=levels)
  return result


def _drop_unused_levels(
    self: pd.DataFrame,
    target_columns: Optional[Union[str, Sequence[str]]] = None) -> pd.DataFrame:
  """Removes unused levels from categorical columns.

  Args:
    self: pandas DataFrame
    target_columns: Name[s] of the column[s] to be have unused levels removed.
      Accepts a string or a list of strings.  If missing, all categorical
      columns have unused levels removed.

  Returns:
    A copy of the input DataFrame with the levels of appropriate columns set to
    those levels present in the column.
  """
  res = self.copy()
  if not target_columns:
    target_columns = res.columns
  target_columns = _adapt_scalar_to_vector(target_columns)
  for c in target_columns:
    if pd.api.types.is_categorical_dtype(res[c]):
      res[c].cat.remove_unused_categories(inplace=True)
  return res


def _reverse_categories(
    self: pd.DataFrame,
    target_columns: Optional[Union[str, Sequence[str]]] = None) -> pd.DataFrame:
  """Reverses the levels of a categorical columns.

  Args:
    self: pandas DataFrame
    target_columns: Name[s] of the column[s] to be reversed.  Accepts a string
      or a list of strings.  If missing, all categorical columns are reversed.

  Returns:
    A copy of the input DataFrame with the levels of appropriate columns
    reversed.
  """
  res = self.copy()
  if not target_columns:
    target_columns = res.columns
  target_columns = _adapt_scalar_to_vector(target_columns)
  for c in target_columns:
    if pd.api.types.is_categorical_dtype(res[c]):
      res.loc[:, c] = pd.Categorical(
          res[c], categories=res[c].values.categories[::-1])
  return res


def _list_values_to_string(input_list: Sequence[Any]) -> Sequence[str]:
  return [str(val) for val in input_list]


def _flatten_columns(self: pd.DataFrame, sep: str = '_') -> pd.DataFrame:
  """Flattens MultiIndex column into Index column concatenated using sep.

  Args:
    self: pandas DataFrame
    sep: separator to use when joining column name strings.

  Returns:
    A DataFrame with a flattened column index.
  """
  if not isinstance(self.columns, pd.MultiIndex):
    return self
  else:
    temp_df = self.copy()
    temp_df.columns = [
        sep.join(_list_values_to_string(col)).strip(sep)
        for col in temp_df.columns
    ]
    return temp_df


def _select_columns(self: pd.DataFrame, reg_ex_list: Union[str, Sequence[str]],
                    **kwargs) -> pd.DataFrame:
  """Selects columns using regular expressions based on column name.

  Args:
    self: pd.DataFrame
    reg_ex_list: A single literal string or regular expression or a list of
      such.
    **kwargs: additional keyword arguments passed to regular expression
      re.compile() e.g. flags=re.IGNORECASE

  Returns:
    A copy of the input DataFrame containing those columns matched by any of the
    regular expressions in reg_ex_list.
  """
  pattern = re.compile(
      '|'.join(
          map(lambda i: '(' + i + ')', _adapt_scalar_to_vector(reg_ex_list))),
      **kwargs)
  selected_cols = [col for col in self.columns if pattern.fullmatch(col)]
  return self[selected_cols].copy()


def _select_rows(
    self: pd.DataFrame, predicate: Callable[[pd.DataFrame],
                                            Sequence[bool]]) -> pd.DataFrame:
  """Select rows of a DataFrame based on a function applied to the frame.

  Args:
    self: input DataFrame
    predicate: A function which takes self as an argument and returns a list of
      booleans of length self.shape[0]

  Returns:
    A copy of the input DataFrame containing those rows corresponding to True
    in predicate(self)
  """
  return self[predicate(self)].copy()


def _groupby_select_rows(
    self: pd.core.groupby.generic.DataFrameGroupBy,
    predicate: Callable[[pd.DataFrame], Sequence[bool]]) -> pd.DataFrame:
  """Calls _select_rows() on each group with the supplied predicate.

  Args:
    self: a grouped pandas DataFrame
    predicate: A function which takes a DataFrame, df, as an argument and
      returns a list of booleans of length df.shape[0]

  Returns:
    A DataFrame obtained by concatenating the results of _select_rows(predicate)
    on each group.
  """
  return self.apply(lambda grp: grp.select_rows(predicate))


def _groupby_assign(self: pd.core.groupby.generic.DataFrameGroupBy,
                    **kwargs) -> pd.DataFrame:
  """Calls _assign() on each group with kwargs passed to each invocation.

  Args:
    self: a groupby DataFrame
    **kwargs: arguments defining assignment, e.g. max_t = lambda df: ...

  Returns:
    A DataFrame obtained by concatenating the results of _assign(**kwargs)
    on each group.
  """
  return self.apply(lambda grp: grp.assign(**kwargs))


def _to_date(self: pd.DataFrame, target_columns: Union[str, Sequence[str]],
             **kwargs) -> pd.DataFrame:
  """Converts specified columns to datetime type with pd.to_datetime().

  Args:
    self: input DataFrame
    target_columns: Name[s] of the column[s] to be coerced to datetime type.
      Accepts a string or a list of strings.
    **kwargs: additional keyword arguments passed to pd.to_datetime(), e.g.
      format='%Y-%m-%d'.

  Returns:
    A copy of the DataFrame with the specified columns converted to datetime
    type.
  """
  res = self.copy()
  for col in _adapt_scalar_to_vector(target_columns):
    res.loc[:, col] = pd.to_datetime(res[col], **kwargs)
  return res


def _tidy_groupby(self, *args, **kwargs):
  if 'group_keys' not in kwargs:
    kwargs['group_keys'] = False
  if 'as_index' not in kwargs:
    kwargs['as_index'] = False
  return self.groupby(*args, **kwargs)


def _equisample(self: pd.DataFrame,
                grouping: Union[str, Sequence[str]],
                n: Optional[int] = None,
                **kwargs) -> pd.DataFrame:
  """Samples the dataframe equally by groups.

  Args:
    self: pandas DataFrame
    grouping: a column name or list of column names specifying the groups to
      sample equally
    n: if present the number of samples to take from each group. If missing the
      size of the smallest group dictates the size of the sample.
    **kwargs: additional arguments passed to pandas dataframe sample method
      including replace= and random_state= to control replacement and
      reproducibility.

  Returns:
    A dataframe with equal sized random samples from each of level of the
    grouping.
  """
  if not n:
    n = self.tidy_groupby(grouping).size()['size'].min()
  return (
      self.tidy_groupby(grouping).apply(lambda grp: grp.sample(n, **kwargs)))


def _tee(self: pd.DataFrame,
         function: Callable[[pd.Series], Any],
         print_result: Optional[bool] = True):
  """Apply a function to a dataframe, printing the result, and return frame.

  Args:
    self: input DataFrame
    function: a function to be applied to the dataframe.
    print_result: whether or not to print results to stdout.

  Returns:
    The input dataframe.
  """
  result = function(self)
  if print_result:
    print(result)
  return self


def _alt_chart(self: pd.DataFrame, **kwargs) -> alt.Chart:
  """Wrapper function that allows to use Altair Chart as part of DataFrame pipe.

  Args:
    self: pandas DataFrame
    **kwargs: Optional keyword arguments for alt.Chart object.

  Returns:
    Altair Chart object.
  """
  return alt.Chart(data=self, **kwargs)


def _ggplot(self: pd.DataFrame, *argv, **kwargs) -> gg.ggplot:
  """Wrapper function to use Plotnine (ggplot) as part of DataFrame pipe.

  Args:
    self: pandas DataFrame
    *argv: positional arguments
    **kwargs: Optional keyword arguments for gg.ggplot() object.

  Returns:
    Plotnine ggplot Chart object.
  """
  return gg.ggplot(self, *argv, **kwargs)


pd.DataFrame.set_categorical = _set_categorical
pd.core.groupby.generic.DataFrameGroupBy.set_categorical = (
    _groupby_set_categorical
)
pd.DataFrame.drop_unused_levels = _drop_unused_levels
pd.DataFrame.reverse_categories = _reverse_categories
pd.DataFrame.flatten_columns = _flatten_columns
pd.DataFrame.select_columns = _select_columns
pd.DataFrame.select_rows = _select_rows
pd.core.groupby.generic.DataFrameGroupBy.assign = _groupby_assign
pd.core.groupby.generic.DataFrameGroupBy.select_rows = _groupby_select_rows
pd.DataFrame.to_date = _to_date
pd.DataFrame.tidy_groupby = _tidy_groupby
pd.DataFrame.equisample = _equisample
pd.DataFrame.tee = _tee
pd.DataFrame.alt_chart = _alt_chart
pd.DataFrame.ggplot = _ggplot
