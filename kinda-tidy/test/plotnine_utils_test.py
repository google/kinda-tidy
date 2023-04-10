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

"""Tests for plotnine_utils."""

import numpy as np
import pandas as pd

from google3.video.youtube.analytics.data_science.analyses.paid_subs.music.lib.python import plotnine_utils
from google3.testing.pybase import googletest


class PlotnineUtilsTest(googletest.TestCase):

  def test_kmgt_formatting(self):
    result_labels = plotnine_utils.kmgt_labels(1.23 *
                                               np.float_power(10., range(28)))
    expected_labels = [
        '1.23', '12.3', '123', '1.23K', '12.3K', '123K', '1.23M', '12.3M',
        '123M', '1.23G', '12.3G', '123G', '1.23T', '12.3T', '123T', '1.23P',
        '12.3P', '123P', '1.23E', '12.3E', '123E', '1.23Z', '12.3Z', '123Z',
        '1.23Y', '12.3Y', '123Y', '1.23e+27'
    ]
    self.assertEqual(result_labels, expected_labels)

  def test_kmgt_formatting_with_prefix(self):
    result_labels = plotnine_utils.kmgt_labels(
        1.23 * np.float_power(10., range(28)), prefix='$')
    expected_labels = [
        '$1.23', '$12.3', '$123', '$1.23K', '$12.3K', '$123K', '$1.23M',
        '$12.3M', '$123M', '$1.23G', '$12.3G', '$123G', '$1.23T', '$12.3T',
        '$123T', '$1.23P', '$12.3P', '$123P', '$1.23E', '$12.3E', '$123E',
        '$1.23Z', '$12.3Z', '$123Z', '$1.23Y', '$12.3Y', '$123Y', '$1.23e+27'
    ]
    self.assertEqual(result_labels, expected_labels)

  def test_kmgt_formatting_with_suffix(self):
    result_labels = plotnine_utils.kmgt_labels(
        1.23 * np.float_power(10., range(28)), suffix='%')
    expected_labels = [
        '1.23%', '12.3%', '123%', '1.23K%', '12.3K%', '123K%', '1.23M%',
        '12.3M%', '123M%', '1.23G%', '12.3G%', '123G%', '1.23T%', '12.3T%',
        '123T%', '1.23P%', '12.3P%', '123P%', '1.23E%', '12.3E%', '123E%',
        '1.23Z%', '12.3Z%', '123Z%', '1.23Y%', '12.3Y%', '123Y%', '1.23e+27%'
    ]
    self.assertEqual(result_labels, expected_labels)

  def test_quarter_labels(self):
    result_labels = plotnine_utils.quarter_labels(
        pd.to_datetime([
            '20211231', '20220101', '2022-04-01 12:00:00', '20220704',
            '20221010'
        ]))
    expected_labels = ['2021Q4', '2022Q1', '2022Q2', '2022Q3', '2022Q4']
    self.assertEqual(result_labels, expected_labels)


if __name__ == '__main__':
  googletest.main()
