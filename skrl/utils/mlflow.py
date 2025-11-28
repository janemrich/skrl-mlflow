import os
from typing import Any, Optional

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


def is_mlflow_artifact(path: str) -> bool:
    return isinstance(path, str) and path.startswith(MLFLOW_ARTIFACT_PREFIX)


def download_mlflow_with_params(artifact_file_uri: str) -> tuple[str, str]:
    """
    Input:
        mlflow-artifacts:/<exp_id>/<run_id>/artifacts/checkpoints/agent_108000.pt

    Output:
        (local_checkpoint_path, local_params_path)

    Local folder structure:
        mlflow-downloads/<run_id>/checkpoints/...
        mlflow-downloads/<run_id>/params/params.yaml
    """

    client = mlflow.MlflowClient()

    # Example:
    # artifact_file_uri =
    #   "mlflow-artifacts:/63/5ba2f4...fe1/artifacts/checkpoints/agent_108000.pt"

    # Strip the "mlflow-artifacts:/" prefix
    no_prefix = artifact_file_uri.replace(MLFLOW_ARTIFACT_PREFIX, "")
    # Now: "63/5ba2f4...fe1/artifacts/checkpoints/agent_108000.pt"

    segments = no_prefix.split("/")
    # segments[0] = experiment_id (e.g. "63")
    # segments[1] = run_id (e.g. "5ba2f4...fe1")
    run_id = segments[1]

    # Find part after ".../artifacts/"
    idx = artifact_file_uri.index("/artifacts/")
    tail = artifact_file_uri[idx + len("/artifacts/"):]
    # tail example: "checkpoints/agent_108000.pt"

    parts = tail.split("/")
    subfolder = parts[0]      # "checkpoints"
    filename = parts[-1]      # "agent_108000.pt"

    # Build local directory structure:
    # mlflow-downloads/<run_id>/checkpoints
    # mlflow-downloads/<run_id>/params
    run_dir = os.path.join("mlflow-downloads", run_id)
    checkpoints_dir = os.path.join(run_dir, "artifacts/checkpoints")
    params_dir = os.path.join(run_dir, "artifacts/params")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)

    # Paths inside MLflow artifacts (relative to the run's artifact root)
    artifact_checkpoint_path = f"{subfolder}/{filename}"      # "checkpoints/agent_108000.pt"
    artifact_params_path = "params/agent.yaml"

    print(1, checkpoints_dir, artifact_checkpoint_path)
    # Download checkpoint
    local_ckpt = client.download_artifacts(
        run_id=run_id,
        path=artifact_checkpoint_path,
        dst_path=checkpoints_dir
    )
    print(2, params_dir, artifact_params_path)

    # Download params.yaml
    try:
        local_params = client.download_artifacts(
            run_id=run_id,
            path="params/agent.yaml",
            dst_path=params_dir
        )

    except mlflow.MlflowException as e:
        print(f"⚠️ Failed to download params.yaml: {e}")
        local_params = None

    return local_ckpt, local_params
