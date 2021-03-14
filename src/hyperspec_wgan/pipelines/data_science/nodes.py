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

"""Node definitions for data science tasks."""

from typing import Any, Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def fit_pca(x: np.ndarray, n_components: float, whiten: bool) -> Dict[str, np.ndarray]:
    """Fit a PCA model to the data."""
    model = PCA(n_components=n_components, whiten=whiten)
    return dict(x=model.fit_transform(X=x), variance=model.explained_variance_ratio_)


def fit_tsne(
    x: np.ndarray,
    perplexity: float,
    early_exaggeration: int,
    learning_rate: float,
    iterations: int,
) -> Any:
    """Fit a t-SNE model to the data."""
    model = TSNE(
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        n_iter=iterations,
        random_state=42,
        n_jobs=-1,
    )
    return model.fit_transform(X=x)
