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

"""Pipeline structure for data science tasks."""

from kedro.pipeline.node import node
from kedro.pipeline.pipeline import Pipeline

from .nodes import fit_pca, fit_tsne


def data_science_pipeline() -> Pipeline:
    """Create the data science pipeline."""
    return Pipeline(
        nodes=[
            node(
                func=fit_pca,
                inputs={
                    "x": "primary_classified_x",
                    "n_components": "params:n_components",
                    "whiten": "params:whiten",
                },
                outputs={
                    "x": "model_output_pca_x",
                    "variance": "model_output_pca_variance",
                },
                name="fit-pca",
                tags="pca",
            ),
            node(
                func=fit_tsne,
                inputs={
                    "x": "primary_classified_x",
                    "perplexity": "params:perplexity",
                    "early_exaggeration": "params:early_exaggeration",
                    "learning_rate": "params:learning_rate",
                    "iterations": "params:iterations",
                },
                outputs="model_output_tsne_x",
                name="fit-tsne",
                tags="tsne",
            ),
        ]
    )
