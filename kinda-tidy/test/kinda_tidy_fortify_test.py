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

"""Tests for kinda_tidy_fortify."""


import numpy as np
import pandas as pd
from pandas import testing as pd_testing
import statsmodels.base.elastic_net
import statsmodels.formula.api as smf

from google3.testing.pybase import googletest
# pylint: disable=unused-import
from google3.video.youtube.analytics.data_science.analyses.paid_subs.music.lib.python import kinda_tidy_fortify
# pylint: enable=unused-import


class KindaTidyFortifyTest(googletest.TestCase):

  # we test with NaNs in predictor and response to ensure that the fitted
  # values are properly aligned even if the NaNs have been removed during
  # model fitting.

  # The implementation of glm fitting in the statsmodel package raises a
  # PerfectSeparationError for the simple test cases so we include one frame
  # with an imperfect fit.
  def setUp(self):
    super().setUp()
    self.df_full = pd.DataFrame(
        {'x': [1.0, 2.0, 3.0, 4.0], 'y': [1.0, 2.0, 3.0, 4.0]}
    )
    self.df_nan_predictor = pd.DataFrame(
        {'x': [1.0, 2.0, np.nan, 4.0], 'y': [1.0, 2.0, 3.0, 4.0]}
    )
    self.df_nan_response = pd.DataFrame(
        {'x': [1.0, 2.0, 3.0, 4.0], 'y': [1.0, 2.0, np.nan, 4.0]}
    )
    self.df_glm = pd.DataFrame(
        {'x': [1.0, 2.0, 3.0, 4.0, 5.0], 'y': [0.0, 0.0, 1.0, 0.0, 1.0]}
    )

  def test_ols(self):
    def fit_model(df):
      return smf.ols(formula='y ~ x', data=df).fit()

    # pyformat: disable
    fortified_full = (self.df_full
                      .pipe(fit_model)
                      .fortify()
                      )
    # pyformat: enable
    expected_full = self.df_full.assign(fitted=[1.0, 2.0, 3.0, 4.0])
    pd_testing.assert_frame_equal(fortified_full, expected_full)

  def test_robust(self):
    def fit_model(df):
      return smf.rlm(formula='y ~ x', data=df).fit()

    # pyformat: disable
    fortified_full = (self.df_full
                      .pipe(fit_model)
                      .fortify()
                      )
    # pyformat: enable
    expected_full = self.df_full.assign(fitted=[1.0, 2.0, 3.0, 4.0])
    pd_testing.assert_frame_equal(fortified_full, expected_full)

  def test_regularized(self):
    def fit_model(df):
      return smf.ols(formula='y ~ x', data=df).fit_regularized()

    # pyformat: disable
    fortified_full = (self.df_full
                      .pipe(fit_model)
                      .fortify()
                      )
    # pyformat: enable
    expected_full = self.df_full.assign(fitted=[1.0, 2.0, 3.0, 4.0])
    pd_testing.assert_frame_equal(fortified_full, expected_full, atol=1.0e-3)

  def test_glm(self):
    def fit_model(df):
      return smf.glm(
          formula='y ~ x', data=df, family=statsmodels.api.families.Binomial()
      ).fit()

    # pyformat: disable
    fortified_full = (self.df_glm
                      .pipe(fit_model)
                      .fortify()
                      )
    # pyformat: enable
    expected_full = self.df_glm.assign(
        fitted=[
            0.05713311658555467,
            0.15276004134289048,
            0.3491698855426487,
            0.6148476385438132,
            0.8260893179850937,
        ]
    )
    pd_testing.assert_frame_equal(fortified_full, expected_full)

  def test_nan_handling(self):
    def fit_model(df):
      return smf.ols(formula='y ~ x', data=df).fit()

    # pyformat: disable
    fortified_nan_predictor = (self.df_nan_predictor
                               .pipe(fit_model)
                               .fortify()
                               )
    # pyformat: enable
    expected_nan_predictor = self.df_nan_predictor.assign(
        fitted=[1.0, 2.0, np.nan, 4.0]
    )
    pd_testing.assert_frame_equal(
        fortified_nan_predictor, expected_nan_predictor
    )

    # pyformat: disable
    fortified_nan_response = (self.df_nan_response
                              .pipe(fit_model)
                              .fortify()
                              )
    # pyformat: enable
    expected_nan_response = self.df_nan_response.assign(
        fitted=[1.0, 2.0, np.nan, 4.0]
    )
    pd_testing.assert_frame_equal(fortified_nan_response, expected_nan_response)


if __name__ == '__main__':
  googletest.main()
