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

"""Custom dataset module for MATLAB files to be used with DataCatalog."""

from pathlib import PurePath
from typing import Any, Dict, Union

import fsspec
from kedro.io.core import AbstractDataSet, get_filepath_str, get_protocol_and_path
from scipy.io import loadmat, savemat


class MatlabDataSet(AbstractDataSet):
    """Load and save data with MATLAB files."""

    def __init__(self, filepath: str) -> None:
        protocol, path = get_protocol_and_path(filepath=filepath)
        self._protocol = protocol
        self._filepath = PurePath(path)
        self._filesystem = fsspec.filesystem(protocol=protocol)

    def _load(self) -> Any:
        filepath = get_filepath_str(path=self._filepath, protocol=self._protocol)
        with self._filesystem.open(path=filepath) as openfile:
            return loadmat(file_name=openfile)

    def _save(self, data: Dict[str, Any]) -> Any:
        filepath = get_filepath_str(path=self._filepath, protocol=self._protocol)
        with self._filesystem.open(path=filepath, mode="wb") as openfile:
            return savemat(file_name=openfile, mdict=data)

    def _describe(self) -> Dict[str, Union[PurePath, str]]:
        return dict(filepath=self._filepath, protocol=self._protocol)
