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

"""Set of helper functions to facilitate plotting with altair."""

from collections.abc import Sequence
import numbers
from typing import Tuple

import altair as alt
import pandas as pd

Number = numbers.Number


def alt_horizontal_line(y_value: float = 0.0,
                        y: str = 'y',
                        color: str = 'black',
                        size: int = 2,
                        **line_kwargs) -> alt.Chart:
  """Utility to plot a horizontal line in Altair within tidyverse lib.

  Args:
    y_value: Value where the horizontal line intersects y-axis.
    y: Name of the column (default should be fine for most cases).
    color: Color of the line, can be a string accepted by Vega lite or HEX.
    size: Width of the line.
    **line_kwargs: Additional keyword arguments passed to mark_rule().

  Returns:
    Altair Chart object with the horizontal line. Can be used within tidyverse
    package like this:
      df.alt_chart().mark(...).encode(...) + alt_horizontal_line().
  """
  temp_data = pd.DataFrame({y: [y_value]})
  line = alt.Chart(temp_data).mark_rule(
      color=color, size=size, **line_kwargs).encode(y=y)
  return line


def alt_vertical_line(x_value: float = 0.0,
                      x: str = 'x',
                      color: str = 'black',
                      size: int = 2,
                      **line_kwargs) -> alt.Chart:
  """Utility to plot a vertical line in Altair within tidyverse lib.

  Args:
    x_value: Value where the vertical line intersects x-axis.
    x: Name of the column (default should be fine for most cases).
    color: Color of the line, can be a string accepted by Vega lite or HEX.
    size: Width of the line.
    **line_kwargs: Additional keyword arguments passed to mark_rule().

  Returns:
    Altair Chart object with the vertical line. Can be used within tidyverse
    package like this:
      df.alt_chart().mark(...).encode(...) + alt_vertical_line().
  """
  temp_data = pd.DataFrame({x: [x_value]})
  line = alt.Chart(temp_data).mark_rule(
      color=color, size=size, **line_kwargs).encode(x=x)
  return line


# pylint: disable-next=dangerous-default-value
def alt_diagonal_line(x_start_end_list: Tuple[float, float] = (0.0, 1.0),
                      y_start_end_list: Tuple[float, float] = (0.0, 1.0),
                      x: str = 'x',
                      y: str = 'y',
                      color: str = 'black',
                      size: int = 2,
                      opacity: float = 0.4,
                      stroke_dash: Sequence[int] = [5, 5],
                      **line_kwargs) -> alt.Chart:
  """Utility to plot a diagonal line in Altair within tidyverse lib.

  Args:
    x_start_end_list: start and end values on x axis; can technically draw any
      arbitrary line by specifying corresponding x and y values.
    y_start_end_list: start and end values on y axis.
    x: Name of the x column (default should be fine for most cases).
    y: Name of the x column (default should be fine for most cases).
    color: Color of the line, can be a string accepted by Vega lite or HEX.
    size: Width of the line.
    opacity: Line opacity.
    stroke_dash: stroke type of the line. Default is [5, 5]: - - - -
    **line_kwargs: Additional keyword arguments passed to mark_rule().

  Returns:
    Altair Chart object with the vertical line. Can be used within tidyverse
    package like this:
      df.alt_chart().mark(...).encode(...) + alt_diagonal_line().
  """
  temp_data = pd.DataFrame({x: x_start_end_list, y: y_start_end_list})
  line = (
      alt.Chart(temp_data).mark_line(
          size=size,
          color=color,
          opacity=opacity,
          strokeDash=stroke_dash,
          **line_kwargs).encode(x=x, y=y))

  return line
