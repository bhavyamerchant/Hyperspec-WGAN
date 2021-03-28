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

from .nodes import extract, reshape, scale, separate


def data_engineering_pipeline() -> Pipeline:
    """Create the data engineering pipeline."""
    return Pipeline(
        nodes=[
            node(
                func=extract,
                inputs="raw_image",
                outputs="intermediate_x",
                name="extract-image",
                tags=["pca", "tsne"],
            ),
            node(
                func=extract,
                inputs="raw_ground_truth",
                outputs="intermediate_y",
                name="extract-ground-truth",
                tags=["pca", "tsne"],
            ),
            node(
                func=reshape,
                inputs={
                    "x": "intermediate_x",
                    "y": "intermediate_y",
                },
                outputs={"x": "reshape_x", "y": "reshape_y"},
                name="reshape-intermediate-dataset",
                tags=["pca", "tsne"],
            ),
            node(
                func=scale,
                inputs="reshape_x",
                outputs="scale_x",
                name="scale-samples",
                tags=["pca", "tsne"],
            ),
            node(
                func=separate,
                inputs={"x": "scale_x", "y": "reshape_y"},
                outputs={
                    "classified_x": "primary_classified_x",
                    "unclassified_x": "primary_unclassified_x",
                    "classified_y": "primary_classified_y",
                    "unclassified_y": "primary_unclassified_y",
                },
                name="separate-data",
                tags=["pca", "tsne"],
            ),
        ]
    )
