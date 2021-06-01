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

"""Node definitions for data visualization tasks."""

from typing import Any, Dict, List

import numpy as np
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure


def _plot_colorbar(figure: Figure, palette: List[str], labels: List[str]) -> None:
    colormap = ListedColormap(colors=palette)
    colorbar = figure.colorbar(mappable=ScalarMappable(cmap=colormap))
    colorbar_range = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks(
        ticks=[
            colorbar.vmin
            + 0.5 * colorbar_range / len(palette)
            + i * colorbar_range / len(palette)
            for i in range(len(palette))
        ]
    )
    colorbar.set_ticklabels(ticklabels=labels)
    colorbar.ax.invert_yaxis()


def plot_pca(
    x: np.ndarray,
    y: np.ndarray,
    variance: np.ndarray,
    metadata: Dict[str, Any],
    kwargs: Dict[str, Any],
) -> sns.FacetGrid:
    """Plot the PCA results."""
    with sns.plotting_context(context="paper"):
        graph = sns.relplot(
            x=x[:, 0],
            y=x[:, 1],
            hue=y,
            palette=metadata["palette"],
            **kwargs["relplot_kwargs"],
        )
        _plot_colorbar(
            figure=graph.fig,
            palette=[*metadata["palette"].values()][1:],
            labels=[*metadata["labels"].values()][1:],
        )
        graph.set(
            title=f'{metadata["name"]} PCA Projection',
            xlabel=f"Principal Component 1 - {variance[0]*100:.1f}% Explained Variance",
            xticks=[],
            ylabel=f"Principal Component 2 - {variance[1]*100:.1f}% Explained Variance",
            yticks=[],
        )
    return graph


def plot_tsne(
    x: np.ndarray,
    y: np.ndarray,
    metadata: Dict[str, Any],
    kwargs: Dict[str, Any],
) -> sns.FacetGrid:
    """Plot the t-SNE results."""
    with sns.plotting_context(context="paper"):
        graph = sns.relplot(
            x=x[:, 0],
            y=x[:, 1],
            hue=y,
            palette=metadata["palette"],
            **kwargs["relplot_kwargs"],
        )
        _plot_colorbar(
            figure=graph.fig,
            palette=[*metadata["palette"].values()][1:],
            labels=[*metadata["labels"].values()][1:],
        )
        graph.set(
            title=f'{metadata["name"]} t-SNE Projection',
            xlabel="t-SNE Component 1",
            xticks=[],
            ylabel="t-SNE Component 2",
            yticks=[],
        )
    return graph
