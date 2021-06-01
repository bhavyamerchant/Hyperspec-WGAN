# Copyright 2021 QuantumBlack Visual Analytics Limited
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

from typing import Dict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import maxabs_scale, minmax_scale, robust_scale
from sklearn.preprocessing import scale as standard_scale


def extract(matlab_data: Dict[str, np.ndarray]) -> np.ndarray:
    """Extract image data from the last value in a `dict`."""
    image = list(matlab_data.values())[-1]
    return image


def scale(image: np.ndarray, kwargs: Dict[str, str]) -> np.ndarray:
    """Scale an image using the desired scaling function."""
    x = np.reshape(a=image, newshape=(-1, image.shape[2]))
    switch = {
        "standard": standard_scale(X=x, **kwargs["standard_scale_kwargs"]),
        "maxabs": maxabs_scale(X=x, **kwargs["maxabs_scale_kwargs"]),
        "minmax": minmax_scale(X=x, **kwargs["minmax_scale_kwargs"]),
        "robust": robust_scale(X=x, **kwargs["robust_scale_kwargs"]),
        "none": x,
    }
    scale_image = np.reshape(a=switch[kwargs["scaler"]], newshape=image.shape)
    return scale_image


def separate(image: np.ndarray, ground_truth: np.ndarray) -> Dict[str, np.ndarray]:
    """Separate classified and unclassified samples."""
    x = np.reshape(a=image, newshape=(-1, image.shape[2]))
    y = np.reshape(a=ground_truth, newshape=-1)
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


def split(
    x: np.ndarray,
    y: np.ndarray,
    kwargs: Dict[str, float],
) -> Dict[str, np.ndarray]:
    """Split the dataset into training, testing, and validation datasets."""
    x_train, x_rest, y_train, y_rest = train_test_split(
        x, y, train_size=kwargs["train_ratio"], random_state=42, stratify=y
    )
    x_test, x_valid, y_test, y_valid = train_test_split(
        x_rest,
        y_rest,
        train_size=(0.1 / (kwargs["test_ratio"] + kwargs["valid_ratio"])),
        random_state=42,
        stratify=y_rest,
    )
    return dict(
        x_train=x_train,
        x_test=x_test,
        x_valid=x_valid,
        y_train=y_train,
        y_test=y_test,
        y_valid=y_valid,
    )
