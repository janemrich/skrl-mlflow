import os
from typing import Any, Optional, Tuple

import mlflow
import socket
import sys


def start_mlflow_run(run_id: Optional[str] = None,
                     experiment_id: Optional[str] = None,
                     run_name: Optional[str] = None,
              nested: bool = False,
              parent_run_id: Optional[str] = None,
              tags: Optional[dict[str, Any]] = None,
              description: Optional[str] = None,
              log_system_metrics: Optional[bool] = None) -> mlflow.ActiveRun:
    """
    Start a new MLflow run, setting it as the active run under which metrics and parameters will be logged

    This function is a wrapper around ``mlflow.start_run`` that adds the following:

    - The command that was called to run the script as the run description
    - A tag 'host' with the hostname of the machine as value

    :param run_id: If specified, get the run with the specified UUID and log parameters and metrics under that run.
                   The run's end time is unset and its status is set to running,
                   but the run's other attributes (source_version, source_type, etc.) are not changed.
    :type run_id: str, optional
    :param experiment_id: ID of the experiment under which to create the current run (applicable only when run_id is not specified).
                          If experiment_id argument is unspecified, will look for valid experiment in the following order:
                          activated using set_experiment, MLFLOW_EXPERIMENT_NAME environment variable,
                          MLFLOW_EXPERIMENT_ID environment variable, or the default experiment as defined by the tracking server.
    :type experiment_id: str, optional
    :param run_name: Name of new run, should be a non-empty string. Used only when run_id is unspecified.
                     If a new run is created and run_name is not specified, a random name will be generated for the run.
    :type run_name: str, optional
    :param nested: Controls whether run is nested in parent run. True creates a nested run.
    :type nested: bool, optional
    :param parent_run_id: If specified, the current run will be nested under the the run with the specified UUID.
                          The parent run must be in the ACTIVE state.
    :type parent_run_id: str, optional
    :param tags: An optional dictionary of string keys and values to set as tags on the run.
                 If a run is being resumed, these tags are set on the resumed run.
                 If a new run is being created, these tags are set on the new run.
    :type tags: dict[str, Any], optional
    :param description: An optional string that populates the description box of the run.
                        If a run is being resumed, the description is set on the resumed run.
                        If a new run is being created, the description is set on the new run.
    :type description: str, optional
    :param log_system_metrics: bool, defaults to None. If True, system metrics will be logged to MLflow, e.g., cpu/gpu utilization.
                               If None, we will check environment variable MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING to determine
                               whether to log system metrics. System metrics logging is an experimental feature in MLflow 2.8 and subject to change.
    :type log_system_metrics: bool, optional

    :return: The MLflow ActiveRun object
    :rtype: mlflow.ActiveRun
    """
    if description is None:
        if sys.platform.startswith("linux"):
            try:
                with open("/proc/self/cmdline", "rb") as f:
                    description = f.read().replace(b"\x00", b" ").strip().decode()
            except Exception:
                description = " ".join(sys.argv)
        else:
            description = " ".join(sys.argv)

    if tags is None:
        tags = {}
    tags["host"] = socket.gethostname()

    return mlflow.start_run(run_id=run_id,
                            experiment_id=experiment_id,
                            run_name=run_name,
                            nested=nested,
                            parent_run_id=parent_run_id,
                            tags=tags,
                            description=description,
                            log_system_metrics=log_system_metrics)


MLFLOW_ARTIFACT_PREFIX = "mlflow-artifacts:/"


def is_mlflow_artifact_uri(uri: str) -> bool:
    """Return True if the string looks like an MLflow artifact URI."""
    return isinstance(uri, str) and uri.startswith(MLFLOW_ARTIFACT_PREFIX)


def _parse_mlflow_artifact_uri(artifact_uri: str) -> Tuple[str, str, str]:
    """
    Parse an MLflow artifact URI.

    Example input:
        mlflow-artifacts:/63/5ba2f4...fe1/artifacts/checkpoints/agent_108000.pt

    Returns:
        (experiment_id, run_id, artifact_rel_path)

        artifact_rel_path is the path **under** the 'artifacts/' root:
            "checkpoints/agent_108000.pt"
    """
    if not is_mlflow_artifact_uri(artifact_uri):
        raise ValueError(f"Not an MLflow artifact URI: {artifact_uri!r}")

    # Strip the scheme prefix
    no_prefix = artifact_uri.replace(MLFLOW_ARTIFACT_PREFIX, "", 1)
    # no_prefix: "<exp_id>/<run_id>/artifacts/checkpoints/agent_108000.pt"

    segments = no_prefix.split("/")
    if len(segments) < 4 or segments[2] != "artifacts":
        raise ValueError(f"Unexpected MLflow artifact URI format: {artifact_uri!r}")

    experiment_id = segments[0]
    run_id = segments[1]
    # everything after ".../artifacts/"
    artifact_rel_path = "/".join(segments[3:])

    return experiment_id, run_id, artifact_rel_path


def download_mlflow_with_params(
    artifact_file_uri: str,
    dst_root: str = "mlflow-downloads",
    params_relative_path: str = "params/agent.yaml",
) -> Tuple[str, Optional[str]]:
    """
    Input:
        artifact_file_uri:
            mlflow-artifacts:/<exp_id>/<run_id>/artifacts/checkpoints/agent_108000.pt

    Output:
        (local_checkpoint_path, local_params_path_or_None)

    Local folder structure (under dst_root):
        <dst_root>/<run_id>/checkpoints/...
        <dst_root>/<run_id>/params/...

    Note: we mirror the *artifact* structure under <dst_root>/<run_id>/.
    """
    _, run_id, artifact_checkpoint_path = _parse_mlflow_artifact_uri(artifact_file_uri)

    client = mlflow.MlflowClient()

    # Base directory for this run’s downloads
    run_dir = os.path.join(dst_root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Download checkpoint
    # This will place the file(s) under:
    #   <dst_root>/<run_id>/<artifact_checkpoint_path>
    local_ckpt = client.download_artifacts(
        run_id=run_id,
        path=artifact_checkpoint_path,
        dst_path=run_dir,
    )

    # Download params (optional)
    local_params: Optional[str] = None
    try:
        local_params = client.download_artifacts(
            run_id=run_id,
            path=params_relative_path,
            dst_path=run_dir,
        )
    except mlflow.MlflowException as e:
        # Soft-fail if there is no params/agent.yaml (or any other error)
        print(f"⚠️ Failed to download {params_relative_path} for run {run_id}: {e}")
        local_params = None

    return local_ckpt, local_params