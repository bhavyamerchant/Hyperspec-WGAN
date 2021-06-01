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

"""Pipeline structure for data engineering tasks."""

from kedro.pipeline.node import node
from kedro.pipeline.pipeline import Pipeline

from .nodes import extract, scale, separate, split


def data_engineering_pipeline() -> Pipeline:
    """Create the data engineering pipeline."""
    return Pipeline(
        nodes=[
            node(
                func=extract,
                inputs="raw_matlab_image",
                outputs="intermediate_image",
                name="extract-image",
                tags=["pca", "tsne", "tcn"],
            ),
            node(
                func=extract,
                inputs="raw_matlab_ground_truth",
                outputs="intermediate_ground_truth",
                name="extract-ground-truth",
                tags=["pca", "tsne", "tcn"],
            ),
            node(
                func=scale,
                inputs={"image": "intermediate_image", "kwargs": "params:scale"},
                outputs="scale_image",
                name="scale-image",
                tags=["pca", "tsne", "tcn"],
            ),
            node(
                func=separate,
                inputs={
                    "image": "scale_image",
                    "ground_truth": "intermediate_ground_truth",
                },
                outputs={
                    "classified_x": "primary_classified_x",
                    "unclassified_x": "primary_unclassified_x",
                    "classified_y": "primary_classified_y",
                    "unclassified_y": "primary_unclassified_y",
                },
                name="separate-classified-and-unclassified-samples",
                tags=["pca", "tsne", "tcn"],
            ),
            node(
                func=split,
                inputs={
                    "x": "primary_classified_x",
                    "y": "primary_classified_y",
                    "kwargs": "params:split",
                },
                outputs={
                    "x_train": "model_input_classified_x_train",
                    "x_test": "model_input_classified_x_test",
                    "x_valid": "model_input_classified_x_valid",
                    "y_train": "model_input_classified_y_train",
                    "y_test": "model_input_classified_y_test",
                    "y_valid": "model_input_classified_y_valid",
                },
                name="split-dataset",
                tags="tcn",
            ),
        ]
    )
