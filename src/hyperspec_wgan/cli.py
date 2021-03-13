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

"""Command line tools intended to be invoked via `kedro`."""

import os
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Union

import click
from kedro.framework.cli.catalog import catalog as catalog_group
from kedro.framework.cli.cli import main as kedro_main
from kedro.framework.cli.pipeline import pipeline as pipeline_group
from kedro.framework.session.session import KedroSession
from kedro.utils import load_obj

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
PROJ_PATH = Path(__file__).resolve().parent

FROM_INPUTS_HELP = """A list of dataset names to be used as a starting point."""
TO_OUTPUTS_HELP = """A list of dataset names which should be used as an end point."""
FROM_NODES_HELP = """A list of node names to be used as a starting point."""
TO_NODES_HELP = """A list of node names to be used as an end point."""
NODE_HELP = """Run only nodes with these specified names."""
TAG_HELP = """Construct the pipeline using only nodes with these specified tags."""
PIPELINE_HELP = """Name of the modular pipeline to run.
If not set, the project pipeline is run by default."""
RUNNER_HELP = """Specify a runner that you want to run the pipeline with.
Available runners: `SequentialRunner`, `ParallelRunner` and `ThreadRunner`."""
ASYNC_HELP = """Load and save node inputs and outputs asynchronously with threads.
If not specified, load and save datasets synchronously."""
ENV_HELP = """Run the pipeline in a configured environment.
If not specified, pipeline will run using environment `local`."""
PARAMS_HELP = """Specify extra parameters that you want to pass
to the context initializer. Items must be separated by comma, keys - by colon,
example: param1:value1,param2:value2. Each parameter is split by the first comma,
so parameter values are allowed to contain colons, parameter keys are not."""
CONFIG_HELP = """Specify a YAML configuration file to load the run
command arguments from. If command line arguments are provided, they will
override the loaded ones."""


def _split_string(
    _ctx: click.Context,
    _param: click.Parameter,
    value: str,
) -> Union[Iterable[str], str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _try_convert_to_numeric(value: Union[float, int, str]) -> Union[float, int, str]:
    try:
        value = float(x=value)
    except ValueError:
        return value
    return int(x=value) if value.is_integer() else value


def _split_params(
    ctx: click.Context, param: click.Parameter, value: str
) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    result = {}
    for item in _split_string(ctx, param, value):
        itemlist = item.split(":", 1)
        if len(itemlist) != 2:
            ctx.fail(
                f"Invalid format of `{param.name}` option: "
                f"Item `{itemlist[0]}` must contain a key and a value separated by `:`."
            )
        key = itemlist[0].strip()
        if not key:
            ctx.fail(
                f"Invalid format of `{param.name}` option: "
                f"Parameter key cannot be an empty string."
            )
        value = itemlist[1].strip()
        result[key] = _try_convert_to_numeric(value)
    return result


def _get_values_as_tuple(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(chain.from_iterable(value.split(",") for value in values))


def _config_callback(
    ctx: click.Context,
    _param: click.Parameter,
    value: click.Path,
) -> click.Path:
    import anyconfig  # pylint: disable=import-outside-toplevel

    if not isinstance(ctx.default_map, dict):
        ctx.default_map = {}
    section = ctx.info_name
    if value:
        config = anyconfig.load(value)[section]
        ctx.default_map.update(config)
    return value


@click.group(context_settings=CONTEXT_SETTINGS, name=__file__)
def cli() -> None:
    """Command line tools for manipulating a Kedro project."""


@cli.command()
@click.option(
    "--from-inputs", type=str, default="", help=FROM_INPUTS_HELP, callback=_split_string
)
@click.option(
    "--to-outputs", type=str, default="", help=TO_OUTPUTS_HELP, callback=_split_string
)
@click.option(
    "--from-nodes", type=str, default="", help=FROM_NODES_HELP, callback=_split_string
)
@click.option(
    "--to-nodes", type=str, default="", help=TO_NODES_HELP, callback=_split_string
)
@click.option("--node", "node_names", type=str, multiple=True, help=NODE_HELP)
@click.option("--tag", type=str, multiple=True, help=TAG_HELP)
@click.option("--pipeline", type=str, default=None, help=PIPELINE_HELP)
@click.option("--runner", type=str, default="SequentialRunner", help=RUNNER_HELP)
@click.option("--async", "is_async", is_flag=True, help=ASYNC_HELP)
@click.option("--env", type=str, default=None, help=ENV_HELP)
@click.option(
    "--params", type=str, default="", help=PARAMS_HELP, callback=_split_params
)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help=CONFIG_HELP,
    callback=_config_callback,
)
def run(
    from_inputs: Iterable[str],
    to_outputs: Iterable[str],
    from_nodes: Iterable[str],
    to_nodes: Iterable[str],
    node_names: Iterable[str],
    tag: Iterable[str],
    pipeline: str,
    runner: str,
    is_async: bool,
    env: str,
    params: Dict[str, Any],
    config: click.Path,  # pylint: disable=unused-argument
) -> None:
    """Run the pipeline."""
    runner_class = load_obj(obj_path=runner, default_obj_path="kedro.runner")
    tag = _get_values_as_tuple(values=tag) if tag else tag
    node_names = _get_values_as_tuple(values=node_names) if node_names else node_names
    package_name = str(Path(__file__).resolve().parent.name)
    with KedroSession.create(
        package_name=package_name, env=env, extra_params=params
    ) as session:
        session.run(
            tags=tag,
            runner=runner_class(is_async=is_async),
            node_names=node_names,
            from_nodes=from_nodes,
            to_nodes=to_nodes,
            from_inputs=from_inputs,
            to_outputs=to_outputs,
            pipeline_name=pipeline,
        )


cli.add_command(catalog_group)
cli.add_command(pipeline_group)

if __name__ == "__main__":
    os.chdir(str(PROJ_PATH))
    kedro_main()
