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

"""Tests for kinda_tidy."""

import io
import re
from unittest import mock

import google3
import altair as alt
import numpy as np
import pandas as pd
from pandas import testing as pd_testing
import plotnine as gg

from google3.testing.pybase import googletest
from google3.video.youtube.analytics.data_science.analyses.paid_subs.music.lib.python import kinda_tidy


class KindaTidyTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.input_dates1 = pd.DataFrame({
        'date1': ['2022-08-08', '2022-08-09 04:04'],
        'date2': ['2022-08-08', '2022-08-09 04:04']
    })
    self.input_dates2 = pd.DataFrame(
        {'date_col': ['18 (Sep) 2022', '19 (Sep) 2022']})

  def test_mean_weighted(self):
    in_values = [1, 2]
    in_wt = [3, 4]
    result = kinda_tidy.mean_weighted(in_values, in_wt)
    expected = (1 * 3 + 2 * 4)/(3 + 4)
    self.assertEqual(expected, result)

  def test_mean_weighted_negvalues(self):
    in_values = [-1, -2]
    in_wt = [3, 4]
    result = kinda_tidy.mean_weighted(in_values, in_wt)
    expected = ((-1) * 3 + (-2) * 4)/(3 + 4)
    self.assertEqual(expected, result)

  def test_mean_weighted_negweights(self):
    in_values = [1, 2]
    in_wt = [-3, 4]
    with self.assertRaises(ValueError):
      _ = kinda_tidy.mean_weighted(in_values, in_wt)

  def test_mean_weighted_difflengths(self):
    in_values = [1, 2]
    in_wt = [3, 4, 5]
    with self.assertRaises(ValueError):
      _ = kinda_tidy.mean_weighted(in_values, in_wt)

  def test_mean_weighted_emptyvalues(self):
    with self.assertRaises(ValueError):
      _ = kinda_tidy.mean_weighted([], [3, 4])

  def test_mean_weighted_emptyweights(self):
    with self.assertRaises(ValueError):
      _ = kinda_tidy.mean_weighted([1, 2], [])

  def test_mean_weighted_nanvalues(self):
    in_values = [1, np.nan]
    in_wt = [3, 4]
    result = kinda_tidy.mean_weighted(in_values, in_wt)
    np.testing.assert_equal(result, np.nan)

  def test_mean_weighted_nanweights(self):
    in_values = [1, 2]
    in_wt = [np.nan, 4]
    result = kinda_tidy.mean_weighted(in_values, in_wt)
    np.testing.assert_equal(result, np.nan)

  def test_mean_weighted_zeroweights(self):
    result = kinda_tidy.mean_weighted([1, 2], [0, 0])
    np.testing.assert_equal(result, np.nan)

  def test_flatten_columns_twolevelindex(self):
    input_df = pd.DataFrame({
        ('col1_level1', 'col1_level2'): [1, 2, 3],
        ('col2_level1', 'col2_level2'): [1, 2, 3]
    })
    result_df = kinda_tidy._flatten_columns(input_df)
    expected_df = pd.DataFrame({
        'col1_level1_col1_level2': [1, 2, 3],
        'col2_level1_col2_level2': [1, 2, 3]
    })
    pd_testing.assert_frame_equal(expected_df, result_df)

  def test_flatten_columns_twolevelindex_diff_separator(self):
    input_df = pd.DataFrame({
        ('col1_level1', 'col1_level2'): [1, 2, 3],
        ('col2_level1', 'col2_level2'): [1, 2, 3]
    })
    result_df = kinda_tidy._flatten_columns(input_df, sep='.')
    expected_df = pd.DataFrame({
        'col1_level1.col1_level2': [1, 2, 3],
        'col2_level1.col2_level2': [1, 2, 3]
    })
    pd_testing.assert_frame_equal(expected_df, result_df)

  def test_flatten_columns_twolevelindex_with_integers(self):
    input_df = pd.DataFrame({
        ('col1_level1', 1): [1, 2, 3],
        ('col2_level1', 2): [1, 2, 3]
    })
    result_df = kinda_tidy._flatten_columns(input_df)
    expected_df = pd.DataFrame({
        'col1_level1_1': [1, 2, 3],
        'col2_level1_2': [1, 2, 3]
    })
    pd_testing.assert_frame_equal(expected_df, result_df)

  def test_flatten_columns_twolevelindex_with_floats(self):
    input_df = pd.DataFrame({
        ('col1_level1', 1.0): [1, 2, 3],
        ('col2_level1', 2.0): [1, 2, 3]
    })
    result_df = kinda_tidy._flatten_columns(input_df)
    expected_df = pd.DataFrame({
        'col1_level1_1.0': [1, 2, 3],
        'col2_level1_2.0': [1, 2, 3]
    })
    pd_testing.assert_frame_equal(expected_df, result_df)

  def test_flatten_columns_threelevelindex(self):
    input_df = pd.DataFrame({
        ('col1_level1', 'col1_level2', 'col1_level3'): [1, 2, 3],
        ('col2_level1', 'col2_level2', 'col2_level3'): [1, 2, 3]
    })
    result_df = kinda_tidy._flatten_columns(input_df)
    expected_df = pd.DataFrame({
        'col1_level1_col1_level2_col1_level3': [1, 2, 3],
        'col2_level1_col2_level2_col2_level3': [1, 2, 3]
    })
    pd_testing.assert_frame_equal(expected_df, result_df)

  def test_flatten_columns_onelevelindex(self):
    input_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [1, 2, 3]})
    result_df = kinda_tidy._flatten_columns(input_df)
    pd_testing.assert_frame_equal(input_df, result_df)

  def test_flatten_columns_intact_input_df(self):
    """Check if the function changes the original DataFrame."""
    input_df = pd.DataFrame({
        ('col1_level1', 'col1_level2'): [1, 2, 3],
        ('col2_level1', 'col2_level2'): [1, 2, 3]
    })
    input_df_copy = pd.DataFrame({
        ('col1_level1', 'col1_level2'): [1, 2, 3],
        ('col2_level1', 'col2_level2'): [1, 2, 3]
    })
    _ = kinda_tidy._flatten_columns(input_df)
    pd_testing.assert_frame_equal(input_df, input_df_copy)

  def test_flatten_columns_as_df_method(self):
    """Check if the function behaves the same if used as a DataFrame method."""
    input_df = pd.DataFrame({
        ('col1_level1', 'col1_level2'): [1, 2, 3],
        ('col2_level1', 'col2_level2'): [1, 2, 3]
    })
    result_df = input_df.flatten_columns()
    expected_df = pd.DataFrame({
        'col1_level1_col1_level2': [1, 2, 3],
        'col2_level1_col2_level2': [1, 2, 3]
    })
    pd_testing.assert_frame_equal(expected_df, result_df)

  def test_flatten_columns_function_idempotence(self):
    """Check if the function behaves the same if used multiple time."""
    input_df = pd.DataFrame({
        ('col1_level1', 'col1_level2'): [1, 2, 3],
        ('col2_level1', 'col2_level2'): [1, 2, 3]
    })
    flatten_once = input_df.flatten_columns()
    flatten_twice = flatten_once.flatten_columns()
    pd_testing.assert_frame_equal(flatten_once, flatten_twice)

  def test_to_date_scalar(self):
    result_df = self.input_dates1.to_date('date1')
    expected_df = pd.DataFrame({
        'date1': pd.to_datetime(['2022-08-08', '2022-08-09 04:04']),
        'date2': ['2022-08-08', '2022-08-09 04:04']
    })
    pd_testing.assert_frame_equal(result_df, expected_df)

  def test_to_date_vector(self):
    result_df = self.input_dates1.to_date(['date1', 'date2'])
    expected_df = pd.DataFrame({
        'date1': pd.to_datetime(['2022-08-08', '2022-08-09 04:04']),
        'date2': pd.to_datetime(['2022-08-08', '2022-08-09 04:04'])
    })
    pd_testing.assert_frame_equal(result_df, expected_df)

  def test_to_date_with_format(self):
    result_df = self.input_dates2.to_date('date_col', format='%d (%b) %Y')
    expected_df = pd.DataFrame({
        'date_col': pd.to_datetime(['2022-09-18', '2022-09-19']),
    })
    pd_testing.assert_frame_equal(result_df, expected_df)

  def test_alt_chart_mark_object(self):
    input_df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3]})
    # Convert chart object to json to make them easy to compare.
    result_alt_object_json = input_df.alt_chart().mark_line().to_json()
    expected_alt_object_json = alt.Chart(input_df).mark_line().to_json()
    self.assertEqual(result_alt_object_json, expected_alt_object_json)

  def test_alt_chart_encoding_object(self):
    input_df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3]})
    # Convert chart object to json to make them easy to compare.
    result_alt_object_json = input_df.alt_chart().mark_line().encode(
        x='x', y='y').to_json()
    expected_alt_object_json = alt.Chart(input_df).mark_line().encode(
        x='x', y='y').to_json()
    self.assertEqual(result_alt_object_json, expected_alt_object_json)

  def test_alt_chart_kwargs(self):
    input_df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3]})
    # Convert chart object to json to make them easy to compare.
    result_alt_object_json = input_df.alt_chart(
        title='chart').mark_line().encode(
            x='x', y='y').to_json()
    expected_alt_object_json = alt.Chart(
        input_df, title='chart').mark_line().encode(
            x='x', y='y').to_json()
    self.assertEqual(result_alt_object_json, expected_alt_object_json)

  def test_ggplot_basic(self):
    input_df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3]})
    expected_ggplot_object = gg.ggplot(input_df, gg.aes(
        x='x', y='y')) + gg.geom_line()
    result_ggplot_object = (
        input_df.ggplot(gg.aes(x='x', y='y')) + gg.geom_line())
    # objects have same methods/variables
    self.assertEqual(dir(result_ggplot_object), dir(expected_ggplot_object))
    # objects have the same underlying data frames
    pd_testing.assert_frame_equal(result_ggplot_object.data,
                                  expected_ggplot_object.data)
    # objects have the same aesthetic mappings
    self.assertEqual(result_ggplot_object.mapping,
                     expected_ggplot_object.mapping)
    # objects have the same types of layers
    self.assertEqual(
        [type(layer.geom) for layer in result_ggplot_object.layers],
        [type(layer.geom) for layer in expected_ggplot_object.layers])


class KindaTidyCategoricalTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.input_df1 = pd.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b', 'c'],
        'value': [1, 2, 3, 4, 5, 7]
    })
    self.input_df2 = pd.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b', 'c'],
        'value': [1, 2, 3, 4, 5, 10]
    })
    self.input_df3 = pd.DataFrame({
        'group1':
            pd.Categorical(['a', 'a', 'a', 'b', 'b', 'c'],
                           categories=['a', 'b', 'c', 'unused']),
        'group2':
            pd.Categorical(['a', 'a', 'a', 'b', 'b', 'c'],
                           categories=['a', 'b', 'c', 'unused']),
        'group3':
            pd.Categorical(['a', 'a', 'a', 'b', 'b', 'c'],
                           categories=['a', 'b', 'c', 'unused']),
        'value': [1, 2, 3, 4, 5, 7]
    })
    self.input_df4 = pd.DataFrame({
        'group1': pd.Categorical(['a', 'a', 'a', 'b', 'b', 'c']),
        'group2': pd.Categorical(['a', 'a', 'a', 'b', 'b', 'c']),
        'group3': pd.Categorical(['a', 'a', 'a', 'b', 'b', 'c']),
        'value': [1, 2, 3, 4, 5, 7]
    })

  def test_default(self):
    result_column = self.input_df1.set_categorical('group').group
    expected_column = pd.Series(
        pd.Categorical(['a', 'a', 'a', 'b', 'b', 'c'],
                       categories=['a', 'b', 'c']),
        name='group')
    pd_testing.assert_series_equal(result_column, expected_column)

  def test_default_aggregation(self):
    result_column = self.input_df1.set_categorical('group', 'value').group
    expected_column = pd.Series(
        pd.Categorical(['a', 'a', 'a', 'b', 'b', 'c'],
                       categories=['b', 'c', 'a']),
        name='group')
    expected_column.cat.categories.name = 'group'
    pd_testing.assert_series_equal(result_column, expected_column)

  def test_sum_aggregator(self):
    result_column = self.input_df1.set_categorical(
        'group', 'value', aggregator=np.sum).group
    expected_column = pd.Series(
        pd.Categorical(['a', 'a', 'a', 'b', 'b', 'c'],
                       categories=['b', 'c', 'a']),
        name='group')
    expected_column.cat.categories.name = 'group'
    pd_testing.assert_series_equal(result_column, expected_column)

  def test_top_n_keep_other(self):
    result_column = self.input_df1.set_categorical(
        'group', 'value', top_n=2, keep_other=True).group
    expected_column = pd.Series(
        pd.Categorical(['_other', '_other', '_other', 'b', 'b', 'c'],
                       categories=['b', 'c', '_other']),
        name='group')
    expected_column.cat.categories.name = 'group'
    pd_testing.assert_series_equal(result_column, expected_column)

  def test_top_n_reverse_keep_other(self):
    result_column = self.input_df1.set_categorical(
        'group', 'value', top_n=2, keep_other=True, reverse=True).group
    expected_column = pd.Series(
        pd.Categorical(['_other', '_other', '_other', 'b', 'b', 'c'],
                       categories=['b', 'c', '_other'][::-1]),
        name='group')
    expected_column.cat.categories.name = 'group'
    pd_testing.assert_series_equal(result_column, expected_column)

  def test_top_n_default(self):
    result_column = self.input_df1.set_categorical(
        'group', 'value', top_n=2).reset_index().group
    expected_column = pd.Series(
        pd.Categorical(['b', 'b', 'c'], categories=['b', 'c']), name='group')
    expected_column.cat.categories.name = 'group'
    pd_testing.assert_series_equal(result_column, expected_column)

  def test_top_n_reverse(self):
    result_column = self.input_df1.set_categorical(
        'group', 'value', top_n=2, reverse=True).reset_index().group
    expected_column = pd.Series(
        pd.Categorical(['b', 'b', 'c'], categories=['b', 'c'][::-1]),
        name='group')
    expected_column.cat.categories.name = 'group'
    pd_testing.assert_series_equal(result_column, expected_column)

  # check that rankings are recalculated with new grouping with other category
  def test_recalculate_order_with_other(self):
    # with aggregator mean c is the largest when we keep 2
    result_column_1 = self.input_df2.set_categorical(
        'group', 'value', top_n=2, keep_other=True,
        aggregator=np.mean).reset_index().group
    expected_column_1 = pd.Series(
        pd.Categorical(['_other', '_other', '_other', 'b', 'b', 'c'],
                       categories=['c', 'b', '_other']),
        name='group')
    expected_column_1.cat.categories.name = 'group'
    pd_testing.assert_series_equal(result_column_1, expected_column_1)
    # If we only keep 1, the _other category gets bigger than c
    result_column_2 = self.input_df2.set_categorical(
        'group', 'value', top_n=1, keep_other=True,
        aggregator=np.sum).reset_index().group
    expected_column_2 = pd.Series(
        pd.Categorical(['_other', '_other', '_other', '_other', '_other', 'c'],
                       categories=['_other', 'c']),
        name='group')
    expected_column_2.cat.categories.name = 'group'
    pd_testing.assert_series_equal(result_column_2, expected_column_2)

  def test_other_label(self):
    result_column = self.input_df1.set_categorical(
        'group', 'value', top_n=2, keep_other=True,
        other_label='new_label').group
    expected_column = pd.Series(
        pd.Categorical(['new_label', 'new_label', 'new_label', 'b', 'b', 'c'],
                       categories=['b', 'c', 'new_label']),
        name='group')
    expected_column.cat.categories.name = 'group'
    pd_testing.assert_series_equal(result_column, expected_column)

  def test_drop_unused_levels_default(self):
    result = self.input_df3.drop_unused_levels()
    contents = ['a', 'a', 'a', 'b', 'b', 'c']
    used_levels = ['a', 'b', 'c']
    expected_column1 = pd.Series(
        pd.Categorical(contents, categories=used_levels))
    expected_column2 = pd.Series(
        pd.Categorical(contents, categories=used_levels))
    expected_column3 = pd.Series(
        pd.Categorical(contents, categories=used_levels))
    pd_testing.assert_series_equal(
        result.group1, expected_column1, check_names=False)
    pd_testing.assert_series_equal(
        result.group2, expected_column2, check_names=False)
    pd_testing.assert_series_equal(
        result.group3, expected_column3, check_names=False)

  def test_drop_unused_levels_scalar(self):
    result = self.input_df3.drop_unused_levels('group2')
    contents = ['a', 'a', 'a', 'b', 'b', 'c']
    original_levels = ['a', 'b', 'c', 'unused']
    used_levels = ['a', 'b', 'c']
    expected_column1 = pd.Series(
        pd.Categorical(contents, categories=original_levels))
    expected_column2 = pd.Series(
        pd.Categorical(contents, categories=used_levels))
    expected_column3 = pd.Series(
        pd.Categorical(contents, categories=original_levels))
    pd_testing.assert_series_equal(
        result.group1, expected_column1, check_names=False)
    pd_testing.assert_series_equal(
        result.group2, expected_column2, check_names=False)
    pd_testing.assert_series_equal(
        result.group3, expected_column3, check_names=False)

  def test_drop_unused_levels_list(self):
    result = self.input_df3.drop_unused_levels(['group2', 'group3'])
    contents = ['a', 'a', 'a', 'b', 'b', 'c']
    original_levels = ['a', 'b', 'c', 'unused']
    used_levels = ['a', 'b', 'c']
    expected_column1 = pd.Series(
        pd.Categorical(contents, categories=original_levels))
    expected_column2 = pd.Series(
        pd.Categorical(contents, categories=used_levels))
    expected_column3 = pd.Series(
        pd.Categorical(contents, categories=used_levels))
    pd_testing.assert_series_equal(
        result.group1, expected_column1, check_names=False)
    pd_testing.assert_series_equal(
        result.group2, expected_column2, check_names=False)
    pd_testing.assert_series_equal(
        result.group3, expected_column3, check_names=False)

  def test_drop_unused_levels_is_idempotent(self):
    apply_once = self.input_df3.drop_unused_levels()
    apply_twice = apply_once.drop_unused_levels()
    pd_testing.assert_frame_equal(apply_once, apply_twice)

  def test_reverse_categories_default(self):
    result = self.input_df4.reverse_categories()
    contents = ['a', 'a', 'a', 'b', 'b', 'c']
    reversed_levels = ['c', 'b', 'a']
    expected_column1 = pd.Series(
        pd.Categorical(contents, categories=reversed_levels))
    expected_column2 = pd.Series(
        pd.Categorical(contents, categories=reversed_levels))
    expected_column3 = pd.Series(
        pd.Categorical(contents, categories=reversed_levels))
    pd_testing.assert_series_equal(
        result.group1, expected_column1, check_names=False)
    pd_testing.assert_series_equal(
        result.group2, expected_column2, check_names=False)
    pd_testing.assert_series_equal(
        result.group3, expected_column3, check_names=False)

  def test_reverse_categories_scalar(self):
    result = self.input_df4.reverse_categories('group2')
    contents = ['a', 'a', 'a', 'b', 'b', 'c']
    original_levels = ['a', 'b', 'c']
    reversed_levels = ['c', 'b', 'a']
    expected_column1 = pd.Series(
        pd.Categorical(contents, categories=original_levels))
    expected_column2 = pd.Series(
        pd.Categorical(contents, categories=reversed_levels))
    expected_column3 = pd.Series(
        pd.Categorical(contents, categories=original_levels))
    pd_testing.assert_series_equal(
        result.group1, expected_column1, check_names=False)
    pd_testing.assert_series_equal(
        result.group2, expected_column2, check_names=False)
    pd_testing.assert_series_equal(
        result.group3, expected_column3, check_names=False)

  def test_reverse_categories_list(self):
    result = self.input_df4.reverse_categories(['group2', 'group3'])
    contents = ['a', 'a', 'a', 'b', 'b', 'c']
    original_levels = ['a', 'b', 'c']
    reversed_levels = ['c', 'b', 'a']
    expected_column1 = pd.Series(
        pd.Categorical(contents, categories=original_levels))
    expected_column2 = pd.Series(
        pd.Categorical(contents, categories=reversed_levels))
    expected_column3 = pd.Series(
        pd.Categorical(contents, categories=reversed_levels))
    pd_testing.assert_series_equal(
        result.group1, expected_column1, check_names=False)
    pd_testing.assert_series_equal(
        result.group2, expected_column2, check_names=False)
    pd_testing.assert_series_equal(
        result.group3, expected_column3, check_names=False)

  def test_reverse_categories_is_self_inverse(self):
    apply_once = self.input_df3.reverse_categories()
    apply_twice = apply_once.reverse_categories()
    pd_testing.assert_frame_equal(self.input_df3, apply_twice)


class KindaTidyGroupbySetCategoricalTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.input_df = pd.DataFrame({
        'continent': [
            'Americas', 'Americas', 'Americas', 'Asia', 'Asia', 'Asia',
            'Europe', 'Europe', 'Europe'
        ],
        'country': [
            'Brazil', 'Mexico', 'United States', 'China', 'India', 'Indonesia',
            'France', 'Germany', 'Turkey'
        ],
        'population': [
            179914212, 102479927, 287675526, 1280400000, 1034172547, 211060000,
            59925035, 82350671, 67308928
        ]
    })

  def test_default(self):
    # pyformat: disable
    result_column = (self.input_df
                     .tidy_groupby('continent')
                     .set_categorical('country', 'population')
                    ).country
    # pyformat: disable
    expected_column = pd.Series(
        pd.Categorical(['Brazil', 'Mexico', 'United States', 'China', 'India', 'Indonesia', 'France', 'Germany', 'Turkey'],
                       categories=['China', 'India', 'Indonesia', 'United States', 'Brazil', 'Mexico',
                                   'Germany', 'Turkey', 'France']), name='country')
    pd_testing.assert_series_equal(result_column, expected_column)

  def test_with_categorical(self):
    # pyformat: disable
    result_column = (self.input_df
                     .set_categorical('continent')
                     .tidy_groupby('continent')
                     .set_categorical('country', 'population')
                    ).country
    # pyformat: disable
    expected_column = pd.Series(
        pd.Categorical(['Brazil', 'Mexico', 'United States', 'China', 'India', 'Indonesia', 'France', 'Germany', 'Turkey'],
                       categories=['China', 'India', 'Indonesia', 'United States', 'Brazil', 'Mexico',
                                   'Germany', 'Turkey', 'France']), name='country')
    pd_testing.assert_series_equal(result_column, expected_column)

  def test_top_2(self):
    # pyformat: disable
    result_column = (self.input_df
                     .tidy_groupby('continent')
                     .set_categorical('country', 'population', top_n=2)
                     .reset_index(drop=True)
                    ).country
    # pyformat: disable
    expected_column = pd.Series(
        pd.Categorical(['Brazil', 'United States', 'China', 'India', 'Germany', 'Turkey'],
                       categories=['China', 'India', 'United States', 'Brazil', 'Germany', 'Turkey']), name='country')
    pd_testing.assert_series_equal(result_column, expected_column)

  def test_top_2_reverse(self):
    # pyformat: disable
    result_column = (self.input_df
                     .tidy_groupby('continent')
                     .set_categorical('country', 'population', top_n=2, reverse=True)
                     .reset_index(drop=True)
                    ).country
    # pyformat: disable
    expected_column = pd.Series(
        pd.Categorical(['Brazil', 'United States', 'China', 'India', 'Germany', 'Turkey'],
                       categories=['China', 'India', 'United States', 'Brazil', 'Germany', 'Turkey'][::-1]), name='country')
    pd_testing.assert_series_equal(result_column, expected_column)

  def test_set_categorical_on_categorical(self):
    # pyformat: disable
    result_column = (self.input_df
                     .set_categorical('country', 'population')
                     .set_categorical('country', 'population', top_n=2)
                     .reset_index(drop=True)
                    ).country
    # pyformat: disable
    expected_column = pd.Series(
        pd.Categorical(['China', 'India'],
                       categories=['China', 'India']), name='country')
    expected_column.cat.categories.names = ['country']
    pd_testing.assert_series_equal(result_column, expected_column)


class KindaTidySelectionTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.input_df = pd.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b', 'c'],
        'value1': [1, 2, 3, 4, 5, 7],
        'value2': [7, 5, 3, 1, -1, -3],
        'walue3': [2, 4, 6, 8, 10, 12]
    })

  def test_select_columns_literal_scalar(self):
    # pyformat: disable
    result_df = (self.input_df
                 .select_columns('value2')
                 .reset_index(drop=True)
                 )
    # pyformat: enable
    expected_df = pd.DataFrame({
        'value2': [7, 5, 3, 1, -1, -3],
    })
    pd_testing.assert_frame_equal(result_df, expected_df)

  def test_select_columns_literal_vector(self):
    # pyformat: disable
    result_df = (self.input_df
                 .select_columns(['group', 'value2'])
                 .reset_index(drop=True)
                 )
    # pyformat: enable
    expected_df = pd.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b', 'c'],
        'value2': [7, 5, 3, 1, -1, -3],
    })
    pd_testing.assert_frame_equal(result_df, expected_df)

  def test_select_columns_pattern_scalar(self):
    # pyformat: disable
    result_df = (self.input_df
                 .select_columns(r'v.*')
                 .reset_index(drop=True)
                 )
    # pyformat: enable
    expected_df = pd.DataFrame({
        'value1': [1, 2, 3, 4, 5, 7],
        'value2': [7, 5, 3, 1, -1, -3],
    })
    pd_testing.assert_frame_equal(result_df, expected_df)

  def test_select_columns_pattern_vector(self):
    # pyformat: disable
    result_df = (self.input_df
                 .select_columns([r'g.*', r'.*3'])
                 .reset_index(drop=True)
                 )
    # pyformat: enable
    expected_df = pd.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b', 'c'],
        'walue3': [2, 4, 6, 8, 10, 12]
    })
    pd_testing.assert_frame_equal(result_df, expected_df)

  def test_select_columns_no_double_selection(self):
    # pyformat: disable
    result_df = (self.input_df
                 .select_columns([r'v.*', r'.*2'])
                 .reset_index(drop=True)
                 )
    # pyformat: enable
    expected_df = pd.DataFrame({
        'value1': [1, 2, 3, 4, 5, 7],
        'value2': [7, 5, 3, 1, -1, -3],
    })
    pd_testing.assert_frame_equal(result_df, expected_df)

  def test_select_columns_pattern_scalar_with_flag(self):
    # pyformat: disable
    result_df = (self.input_df
                 .select_columns(r'V.*', flags=re.IGNORECASE)
                 .reset_index(drop=True)
                 )
    # pyformat: enable
    expected_df = pd.DataFrame({
        'value1': [1, 2, 3, 4, 5, 7],
        'value2': [7, 5, 3, 1, -1, -3],
    })
    pd_testing.assert_frame_equal(result_df, expected_df)

  def test_select_rows_by_one_column(self):
    # pyformat: disable
    result_df = (self.input_df
                 .select_rows(lambda df: df.value1 >= 4)
                 .reset_index(drop=True))
    # pyformat: enable
    expected_df = pd.DataFrame({
        'group': ['b', 'b', 'c'],
        'value1': [4, 5, 7],
        'value2': [1, -1, -3],
        'walue3': [8, 10, 12]
    })
    pd_testing.assert_frame_equal(result_df, expected_df)

  def test_select_rows_by_two_columns(self):
    # pyformat: disable
    result_df = (self.input_df
                 .select_rows(lambda df: df.value1 < df.value2)
                 .reset_index(drop=True))
    # pyformat: enable
    expected_df = pd.DataFrame({
        'group': ['a', 'a'],
        'value1': [1, 2],
        'value2': [7, 5],
        'walue3': [2, 4]
    })
    pd_testing.assert_frame_equal(result_df, expected_df)

  def test_select_rows_with_aggregate(self):
    # pyformat: disable
    result_df = (self.input_df
                 .select_rows(lambda df: df.value1 <= np.mean(df.value1))
                 .reset_index(drop=True))
    # pyformat: enable
    expected_df = pd.DataFrame({
        'group': ['a', 'a', 'a'],
        'value1': [1, 2, 3],
        'value2': [7, 5, 3],
        'walue3': [2, 4, 6]
    })
    pd_testing.assert_frame_equal(result_df, expected_df)

  # In the following we had to drop the index after the groupby because the
  # default index inherits row numbers from the original dataframe.  The option
  # check_index for assert_frame_equal is not currently implemented so we
  # get the same effect by dropping the default index on result_df so that it
  # matches the expected index.
  def test_grouped_select_rows_with_aggregate(self):
    # pyformat: disable
    result_df = (self.input_df
                 .tidy_groupby('group')
                 .select_rows(lambda df: df.value1 <= np.mean(df.value1))
                 .reset_index(drop=True))
    # pyformat: enable
    expected_df = pd.DataFrame({
        'group': ['a', 'a', 'b', 'c'],
        'value1': [1, 2, 4, 7],
        'value2': [7, 5, 1, -3],
        'walue3': [2, 4, 8, 12]
    })
    pd_testing.assert_frame_equal(result_df, expected_df)


class KindaTidyGroupbyTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.input_df = pd.DataFrame({
        'group1': ['a', 'a', 'a', 'b', 'b', 'c'],
        'group2': [1, 1, 2, 2, 2, 2],
        'value': [1, 2, 3, 4, 5, 7],
    })

  def test_single_group_assign(self):
    # pyformat: disable
    result_df = (self.input_df
                 .tidy_groupby('group1')
                 .assign(centered=lambda df: df.value/np.sum(df.value))
                 )
    # pyformat: enable
    expected_df = pd.DataFrame({
        'group1': ['a', 'a', 'a', 'b', 'b', 'c'],
        'group2': [1, 1, 2, 2, 2, 2],
        'value': [1, 2, 3, 4, 5, 7],
        'centered': [
            1 / (1 + 2 + 3), 2 / (1 + 2 + 3), 3 / (1 + 2 + 3), 4 / (4 + 5),
            5 / (4 + 5), 7 / 7
        ],
    })
    pd_testing.assert_frame_equal(result_df, expected_df)

  def test_two_groups_assign(self):
    # pyformat: disable
    result_df = (self.input_df
                 .tidy_groupby(['group1', 'group2'])
                 .assign(centered=lambda df: df.value/np.sum(df.value))
                 )
    # pyformat: enable
    expected_df = pd.DataFrame({
        'group1': ['a', 'a', 'a', 'b', 'b', 'c'],
        'group2': [1, 1, 2, 2, 2, 2],
        'value': [1, 2, 3, 4, 5, 7],
        'centered': [
            1 / (1 + 2), 2 / (1 + 2), 3 / 3, 4 / (4 + 5), 5 / (4 + 5), 7 / 7
        ],
    })
    pd_testing.assert_frame_equal(result_df, expected_df)

  def test_aggregate(self):
    # pyformat: disable
    result_df = (self.input_df
                 .tidy_groupby('group1')
                 .agg({'value': np.sum})
                 )
    # pyformat: enable
    expected_df = pd.DataFrame({
        'group1': ['a', 'b', 'c'],
        'value': [1 + 2 + 3, 4 + 5, 7],
    })
    pd_testing.assert_frame_equal(result_df, expected_df)


class KindaTidyEquisampleTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.random_state = 20220106
    self.input_df = pd.DataFrame({
        'l1': ['a', 'a', 'a', 'a', 'b', 'b', 'b'],
        'l2': ['c', 'c', 'd', 'd', 'c', 'c', 'd'],
        'value': [0, 1, 2, 3, 4, 5, 6]
    })

  def test_default(self):
    # pyformat: disable
    result_df = (self.input_df
                 .equisample('l1', random_state=self.random_state)
                 .reset_index(drop=True)
                )
    # pyformat: enable
    expected_df = pd.DataFrame({
        'l1': ['a', 'a', 'a', 'b', 'b', 'b'],
        'l2': ['d', 'd', 'c', 'd', 'c', 'c'],
        'value': [2, 3, 0, 6, 4, 5]
    })
    pd_testing.assert_frame_equal(result_df, expected_df)

  def test_fixed_size(self):
    # pyformat: disable
    result_df = (self.input_df
                 .equisample('l1', n=2, random_state=self.random_state)
                 .reset_index(drop=True)
                )
    # pyformat: enable
    expected_df = pd.DataFrame({
        'l1': ['a', 'a', 'b', 'b'],
        'l2': ['d', 'd', 'd', 'c'],
        'value': [2, 3, 6, 4]
    })
    pd_testing.assert_frame_equal(result_df, expected_df)

  def test_two_level(self):
    # pyformat: disable
    result_df = (self.input_df
                 .equisample(['l1', 'l2'], random_state=self.random_state)
                 .reset_index(drop=True)
                )
    # pyformat: enable
    expected_df = pd.DataFrame({
        'l1': ['a', 'a', 'b', 'b'],
        'l2': ['c', 'd', 'c', 'd'],
        'value': [0, 2, 4, 6]
    })
    pd_testing.assert_frame_equal(result_df, expected_df)


class TeeTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.input_df = pd.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b', 'c'],
        'value': [1, 2, 3, 4, 5, 7]
    })

  def test_tee(self):
    mock_stdout = io.StringIO()
    with mock.patch('sys.stdout', mock_stdout):
      result_df = (
          self.input_df.tee(lambda df: df.tidy_groupby('group').agg(np.sum)))
    expected_stdout = """  group  value
0     a      6
1     b      9
2     c      7
"""
    pd_testing.assert_frame_equal(result_df, self.input_df)
    self.assertEqual(mock_stdout.getvalue(), expected_stdout)


if __name__ == '__main__':
  googletest.main()
