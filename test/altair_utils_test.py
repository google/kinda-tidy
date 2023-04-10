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

"""Tests for altair_helpers."""

import altair as alt
import pandas as pd

from google3.testing.pybase import googletest
import google3.video.youtube.analytics.data_science.analyses.paid_subs.music.lib.python.altair_utils as alt_utils


class AltairHelpersTest(googletest.TestCase):

  def test_alt_horizontal_line_object(self):
    # Check if the horizontal line objects match up on their own.
    input_df = pd.DataFrame({'y': [0.0]})
    result_alt_object_json = alt_utils.alt_horizontal_line().to_json()
    expected_alt_object_json = alt.Chart(input_df).mark_rule(
        color='black', size=2).encode(y='y').to_json()
    self.assertEqual(result_alt_object_json, expected_alt_object_json)

  def test_alt_horizontal_line_part_of_another_chart(self):
    # Check if the horizontal line objects match up on top of another chart.
    main_df = pd.DataFrame({'y': [-3, 0, 3], 'x': [0, 1, 2]})
    df_for_horizontal_line = pd.DataFrame({'y': [0.0]})
    main_chart = alt.Chart(main_df).mark_line().encode(y='y', x='x')
    expected_horizontal_line = alt.Chart(df_for_horizontal_line).mark_rule(
        color='black', size=2).encode(y='y')
    result_alt_object_json = (main_chart +
                              alt_utils.alt_horizontal_line()).to_json()
    expected_alt_object_json = (main_chart + expected_horizontal_line).to_json()
    self.assertEqual(result_alt_object_json, expected_alt_object_json)

  def test_alt_horizontal_line_kwargs(self):
    # Test custom line kwargs.
    input_df = pd.DataFrame({'y': [0.0]})
    result_alt_object_json = alt_utils.alt_horizontal_line(
        color='red', size=5, strokeDash=[2, 2]).to_json()
    expected_horizontal_line = alt.Chart(input_df).mark_rule(
        color='red', size=5, strokeDash=[2, 2]).encode(y='y').to_json()
    self.assertEqual(result_alt_object_json, expected_horizontal_line)

  def test_alt_vertical_line_object(self):
    # Check if the vertical line objects match up on their own.
    input_df = pd.DataFrame({'x': [0.0]})
    result_alt_object_json = alt_utils.alt_vertical_line().to_json()
    expected_alt_object_json = alt.Chart(input_df).mark_rule(
        color='black', size=2).encode(x='x').to_json()
    self.assertEqual(result_alt_object_json, expected_alt_object_json)

  def test_alt_vertical_line_part_of_another_chart(self):
    # Check if the vertical line objects match up on top of another chart.
    main_df = pd.DataFrame({'y': [-3, 0, 3], 'x': [0, 1, 2]})
    df_for_vertical_line = pd.DataFrame({'x': [0.0]})
    main_chart = alt.Chart(main_df).mark_line().encode(y='y', x='x')
    expected_vertical_line = alt.Chart(df_for_vertical_line).mark_rule(
        color='black', size=2).encode(x='x')
    result_alt_object_json = (main_chart +
                              alt_utils.alt_vertical_line()).to_json()
    expected_alt_object_json = (main_chart + expected_vertical_line).to_json()
    self.assertEqual(result_alt_object_json, expected_alt_object_json)

  def test_alt_vertical_line_kwargs(self):
    # Test custom line kwargs.
    input_df = pd.DataFrame({'x': [0.0]})
    result_alt_object_json = alt_utils.alt_vertical_line(
        color='red', size=5, strokeDash=[2, 2]).to_json()
    expected_horizontal_line = alt.Chart(input_df).mark_rule(
        color='red', size=5, strokeDash=[2, 2]).encode(x='x').to_json()
    self.assertEqual(result_alt_object_json, expected_horizontal_line)

  def test_alt_diagonal_line_object(self):
    # Check if the diagonal line objects match up on their own.
    df_for_diagonal_line = pd.DataFrame({'y': [0.0, 1.0], 'x': [0.0, 1.0]})
    result_alt_object_json = alt_utils.alt_diagonal_line().to_json()
    expected_alt_object_json = alt.Chart(df_for_diagonal_line).mark_line(
        color='black', size=2, opacity=0.4, strokeDash=[5, 5]).encode(
            x='x', y='y').to_json()
    self.assertEqual(result_alt_object_json, expected_alt_object_json)

  def test_alt_diagonal_line_part_of_another_chart(self):
    # Check if the diagonal line objects match up on top of another chart.
    main_df = pd.DataFrame({'y': [0, 0.8, 1], 'x': [0, 0.5, 1]})
    df_for_diagonal_line = pd.DataFrame({'y': [0.0, 1.0], 'x': [0.0, 1.0]})
    main_chart = alt.Chart(main_df).mark_line().encode(y='y', x='x')
    expected_diagonal_line = alt.Chart(df_for_diagonal_line).mark_line(
        color='black', size=2, opacity=0.4, strokeDash=[5, 5]).encode(
            x='x', y='y')
    result_alt_object_json = (main_chart +
                              alt_utils.alt_diagonal_line()).to_json()
    expected_alt_object_json = (main_chart + expected_diagonal_line).to_json()
    self.assertEqual(result_alt_object_json, expected_alt_object_json)

  def test_alt_diagonal_line_kwargs(self):
    # Test custom line kwargs.
    df_for_diagonal_line = pd.DataFrame({'y': [0.0, 1.0], 'x': [0.0, 1.0]})
    result_alt_object_json = alt_utils.alt_diagonal_line(
        color='red', size=5, opacity=1.0, stroke_dash=[2, 2]).to_json()
    expected_horizontal_line = alt.Chart(df_for_diagonal_line).mark_line(
        color='red', size=5, opacity=1.0, strokeDash=[2, 2]).encode(
            x='x', y='y').to_json()
    self.assertEqual(result_alt_object_json, expected_horizontal_line)

  def test_alt_diagonal_line_diff_start_end_values(self):
    # Check if the diagonal line objects match up on their own.
    df_for_diagonal_line = pd.DataFrame({'y': [-1.0, 0.0], 'x': [-1.0, 0.0]})
    result_alt_object_json = alt_utils.alt_diagonal_line(
        x_start_end_list=(-1.0, 0.0), y_start_end_list=(-1.0, 0.0)).to_json()
    expected_alt_object_json = alt.Chart(df_for_diagonal_line).mark_line(
        color='black', size=2, opacity=0.4, strokeDash=[5, 5]).encode(
            x='x', y='y').to_json()
    self.assertEqual(result_alt_object_json, expected_alt_object_json)


if __name__ == '__main__':
  googletest.main()
