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

"""Utilities for plotnine.

Utility functions to make plotnine more useful.
Formatting functions:
  kmgt_labels: converts tick labes from 123,456,789 to 1.23M
  quarter_labels: formats date tick labels to form 2022Q3
Exporting functions:
  save_ggplot_to_drive: exports plot to png in drive
"""

# CodeHealthStats Testing: L1 (mostly tested)
# CodeHealthStats LongestFunction: 20 lines

import datetime
import io
import numbers
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotnine

from google3.research.colab.lib import drive


Number = numbers.Number


def kmgt_labels(numlist: Sequence[Number],
                prefix: str = '',
                suffix: str = '') -> Sequence[str]:
  """Formats ggplot labels with engineering orders of magnitude.

  Args:
    numlist: a list of numerics to be given labels
    prefix: a string prepended to each label e.g. prefix='$'
    suffix: a string appended to each label

  Returns:
    A list of strings of the form d.d[KMGTPEZY]

  Usage:
    In a ggplot axis scaling layer as
    + scale_x_continuous(labels = kmgt_labels)
  """

  def label_one(num):
    if num == 0:
      return '0'
    mille_scale = int(np.floor((np.ceil(np.log10(abs(num * 1.1))) - 1) / 3))
    if mille_scale > 8:
      return prefix + str(num) + suffix
    power_map = {
        0: '',
        1: 'K',
        2: 'M',
        3: 'G',
        4: 'T',
        5: 'P',
        6: 'E',
        7: 'Z',
        8: 'Y'
    }
    return prefix + '{:.3g}'.format(
        num / 10**(3 * mille_scale)) + power_map[mille_scale] + suffix

  return [label_one(n) for n in numlist]


def quarter_labels(date_list: Sequence[datetime.datetime]) -> Sequence[str]:
  """Formats ggplot datetimes in the form ddddQd (2022Q3 eg).

  Args:
    date_list: a list of datetime to be given string labels.

  Returns:
    A list of strings of the form ddddQd

  Usage:
    In a ggplot axis scaling layer as
    + scale_x_date(breaks = ..., labels=quarter_labels)
  """

  def strftimeq(dt):
    return str(dt.year) + 'Q' + str(dt.quarter)

  return [strftimeq(pd.Timestamp(x)) for x in date_list]


def save_ggplot_to_drive(plot_object: plotnine.ggplot,
                         file_name: str,
                         parent_dir: Optional[str] = None):
  """Saves a ggplot object as png to drive.

  Args:
    plot_object: the ggplot object to be saved
    file_name: the name of the file to save to in drive
    parent_dir: Optional, the name of the parent directory
  Notes: The plot_object is converted to png and saved in the given file name in
    the user's drive.  If a parent directory name is specified that directory is
    created (if needed) and the new file is created under that directory. If the
    plot file name already exists it is overwritten by the new png file.
  """
  if parent_dir:
    parent_id = drive.GetOrCreateFolder(parent_dir)
  if not drive.FileExists(file_name):
    drive.CreateFile(
        mime_type='image/png',
        filename=file_name,
        parents=[parent_id] if parent_dir else None)
  byte_stream = io.BytesIO()
  plotnine.ggsave(filename=byte_stream, plot=plot_object)
  drive.SaveFile(
      filename=file_name, data=byte_stream.getbuffer(), mime_type='image/png')
