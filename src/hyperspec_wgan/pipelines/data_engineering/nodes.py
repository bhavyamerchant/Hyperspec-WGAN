# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Node definitions for data engineering tasks."""

from typing import Any, Dict

import numpy as np
from sklearn.preprocessing import maxabs_scale


def extract(data: Dict[str, np.ndarray]) -> np.ndarray:
    """Extract NumPy data from a `dict`."""
    return list(data.values())[-1]


def reshape(x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """Reshape the samples into a 2-D array and the classifications into a 1-D array."""
    x = np.reshape(a=x, newshape=(-1, x.shape[2]))
    y = np.reshape(a=y, newshape=-1)
    return dict(x=x, y=y)


def scale(x: np.ndarray) -> Any:
    """Scale samples to the [-1, 1] range without breaking the sparsity."""
    return maxabs_scale(X=x)


def separate(x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """Separate classified and unclassified samples within the dataset."""
    classified_x = x[y != 0, :]
    unclassified_x = x[y == 0, :]
    classified_y = y[y != 0]
    unclassified_y = y[y == 0]
    return dict(
        classified_x=classified_x,
        unclassified_x=unclassified_x,
        classified_y=classified_y,
        unclassified_y=unclassified_y,
    )
