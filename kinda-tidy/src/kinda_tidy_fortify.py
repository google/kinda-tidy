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

"""Fortify routines for kinda-tidy package.

When including a modeling stage in a kinda-tidy pipeline the results are
typically not pandas dataframes.  The fortify() method will take a fitted
model and return the original dataframe fortified with the fitted values from
the model.  Since each model fitting package can have different ways of
accessing the original dataframe and the fitted values we may need multiple
implementations.
"""


import statsmodels
import statsmodels.api
import statsmodels.base.elastic_net


# The following fortifies models fit by statsmodels


def _fortify_statsmodels(self):
  """Build a dataframe with modeling frame fortified with fitted values."""
  data = self.model.data.frame.copy()
  data['fitted'] = self.fittedvalues
  return data


statsmodels.base.wrapper.ResultsWrapper.fortify = _fortify_statsmodels
