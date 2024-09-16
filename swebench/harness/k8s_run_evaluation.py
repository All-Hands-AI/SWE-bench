from __future__ import annotations

import os
import pathlib
import gzip
import glob
import time
import json
import tempfile
import threading
import traceback
from kubernetes import client, config
from kubernetes.stream import stream
from kubernetes.client.models import V1Pod, V1PodSpec, V1Container, V1ObjectMeta, V1PodSecurityContext
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from kubernetes.client.api import core_v1_api
from io import BytesIO
from google.cloud import storage
import hashlib
import io
import tarfile

from swebench.harness.run_evaluation import (
    get_dataset_from_preds,
    make_run_report,
    load_swebench_dataset,
    get_gold_predictions,
    make_test_spec,
    setup_logger,
    close_logger,
    get_eval_report,
    EvaluationError,
    KEY_INSTANCE_ID,
    APPLY_PATCH_PASS,
    APPLY_PATCH_FAIL,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec import make_test_spec, TestSpec
from swebench.harness.utils import load_swebench_dataset, str2bool

K8S_EXECUTOR_NAMESPACE = "swe-bench-eval-executors"
DOCKER_IMAGE_PREFIX = os.environ.get('EVAL_DOCKER_IMAGE_PREFIX', 'us-docker.pkg.dev/evaluation-428620/swe-bench-images')
print(f'Using docker image prefix: {DOCKER_IMAGE_PREFIX}')

EVAL_ID = os.environ.get('EVAL_ID')
assert EVAL_ID is not None, 'EVAL_ID environment variable is required'
print(f'Getting eval id: {EVAL_ID}')

def get_instance_docker_image(instance_id: str) -> str:
    image_name = 'sweb.eval.x86_64.' + instance_id
    image_name = image_name.replace(
        '__', '_s_'
    )  # to comply with docker image naming convention
    return DOCKER_IMAGE_PREFIX.rstrip('/') + '/' + image_name


def write_file_to_k8s_pod(
    k8s_api: client.CoreV1Api,
    pod_name: str,
    namespace: str,
    file_content: str,
    file_path: str,
    logger=None,
    max_retries=3,
    retry_delay=5
):
    """
    Write file content to a specified path in a Kubernetes pod using tar archiving.

    Args:
        k8s_api (kubernetes.client.CoreV1Api): Kubernetes API client
        pod_name (str): Name of the target pod
        namespace (str): Namespace of the target pod
        file_content (str): Content to write to the file
        file_path (str): Path where the file should be written in the pod
        logger (logging.Logger, optional): Logger for output messages
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Delay between retries in seconds

    Returns:
        bool: True if successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            # Create a temporary file with the content
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file_path = pathlib.Path(temp_file.name)

            # Use the copy_file function to transfer the file to the pod
            copy_file(k8s_api, namespace, pod_name, temp_file_path, file_path)

            if logger:
                logger.info(f"File successfully written to pod at {file_path}")
            return True

        except Exception as e:
            if logger:
                logger.error(f"Error when writing file to pod: {str(e)}")
            if attempt < max_retries - 1:
                if logger:
                    logger.info(f"Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                if logger:
                    logger.error(f"Failed to write file to pod after {max_retries} attempts")
                return False

        finally:
            # Clean up the temporary file
            if 'temp_file_path' in locals():
                os.unlink(temp_file_path)

    return False

def copy_file(kube_conn, namespace: str, pod_name: str, source_file: pathlib.Path, dest_path: str):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:tar') as tar:
        tar.add(source_file, arcname=pathlib.Path(dest_path).name)
    commands = [buf.getvalue()]

    # Copying file
    exec_command = ['tar', 'xvf', '-', '-C', os.path.dirname(dest_path)]
    resp = stream(kube_conn.connect_get_namespaced_pod_exec, pod_name, namespace,
                  command=exec_command,
                  stderr=True, stdin=True,
                  stdout=True, tty=False,
                  _preload_content=False)

    while resp.is_open():
        resp.update(timeout=1)
        if resp.peek_stdout():
            print(f"STDOUT: {resp.read_stdout()}")
        if resp.peek_stderr():
            print(f"STDERR: {resp.read_stderr()}")
        if commands:
            c = commands.pop(0)
            resp.write_stdin(c)
        else:
            break
    resp.close()

def k8s_exec_run_with_timeout(
    api_instance: client.CoreV1Api,
    pod_name: str,
    namespace: str,
    command: str,
    timeout: int | None=60,
):
    """
    Run a command in a Kubernetes pod with a timeout.

    Args:
        api_instance (kubernetes.client.CoreV1Api): Kubernetes API instance.
        pod_name (str): Name of the pod to run the command in.
        namespace (str): Namespace of the pod.
        command (str): Command to run.
        timeout (int): Timeout in seconds.
    """
    exec_result = ''
    timed_out = False

    exec_command = ['/bin/sh', '-c', command]
    resp = stream(api_instance.connect_get_namespaced_pod_exec,
                  pod_name,
                  namespace,
                  command=exec_command,
                  stderr=True, stdin=False,
                  stdout=True, tty=False,
                  _preload_content=False)

    start_time = time.time()
    while resp.is_open():
        resp.update(timeout=1)
        if resp.peek_stdout():
            exec_result += resp.read_stdout()
        if resp.peek_stderr():
            exec_result += resp.read_stderr()
        
        if timeout and (time.time() - start_time > timeout):
            timed_out = True
            break

    if not timed_out:
        resp.close()

    end_time = time.time()
    return exec_result, timed_out, end_time - start_time


def generate_short_name(instance_id: str, eval_id: str, max_length: int = 63) -> str:
    # Create a hash of the EVAL_ID
    hash_object = hashlib.md5(eval_id.encode())
    hash_hex = hash_object.hexdigest()
    
    # Use the first 8 characters of the hash
    short_hash = hash_hex[:8]
    
    # Create a name using instance_id and the short hash of EVAL_ID
    name = f"{instance_id}-{short_hash}"
    
    # Ensure the name is not longer than max_length
    if len(name) > max_length:
        # If it's too long, truncate the instance_id part
        truncated_instance_id = instance_id[:max_length - len(short_hash) - 1]
        name = f"{truncated_instance_id}-{short_hash}"
    
    return name.replace('_', '-').lower()

def k8s_run_instance(
    test_spec: TestSpec,
    pred: dict,
    k8s_api: client.CoreV1Api,
    timeout: int | None = None,
    output_dir: str | None = None,
) -> dict:
    instance_id = test_spec.instance_id
    log_dir = os.path.join(output_dir, 'instances', instance_id)
    os.makedirs(log_dir, exist_ok=True)
    log_file = pathlib.Path(os.path.join(log_dir, "run_instance.log"))
    report_path = pathlib.Path(os.path.join(log_dir, "report.json"))

    if report_path.exists():
        with open(report_path, 'r') as f:
            return instance_id, json.load(f)
    logger = setup_logger(instance_id, log_file)

    eval_name = generate_short_name(instance_id, EVAL_ID)
    try:
        # Create pod using Google Cloud's schema
        pod = V1Pod(
            metadata=V1ObjectMeta(
                name=eval_name,
                labels={
                    "app": "swebench-eval",
                    "instance_id": instance_id,
                    "eval_id": EVAL_ID
                }
            ),
            spec=V1PodSpec(
                containers=[
                    V1Container(
                        name=eval_name,
                        image=get_instance_docker_image(instance_id),
                        command=["/bin/bash", "-c"],
                        args=["while true; do sleep 30; done;"],  # Keep container running
                        working_dir="/workspace",  # Set working directory
                    )
                ],
                restart_policy="Never",
                security_context=V1PodSecurityContext(
                    run_as_user=1000,  # Set user ID
                    run_as_group=1000,  # Set group ID
                    fs_group=1000,  # Set filesystem group
                )
            )
        )
        try:
            pod = k8s_api.create_namespaced_pod(namespace=K8S_EXECUTOR_NAMESPACE, body=pod)
            print(f"[{instance_id}] [info] Pod created: {pod.metadata.name}")
        except Exception as e:
            print(f"[{instance_id}] [error] Failed to create pod: {str(e)}")
            raise

        # Wait for pod to be running
        logger.info(f"[{instance_id}] Waiting for pod to be running...")
        while True:
            try:
                pod = k8s_api.read_namespaced_pod(name=pod.metadata.name, namespace=K8S_EXECUTOR_NAMESPACE)
                logger.info(f"Pod status: {pod.status.phase}")
                if pod.status.phase == "Running":
                    break
            except Exception as e:
                logger.error(f"Error checking pod status: {str(e)}")
                logger.exception(e)
                raise
            time.sleep(1)

        # Write patch file to pod
        logger.info(f"Writing patch file to pod: {pod.metadata.name}")
        patch_content = pred["model_patch"] or ""
        success = write_file_to_k8s_pod(k8s_api, pod.metadata.name, K8S_EXECUTOR_NAMESPACE, 
                                        patch_content, "/tmp/patch.diff", logger)
        if not success:
            raise EvaluationError(instance_id, "Failed to write patch file to pod", logger)
        logger.info(f"[{instance_id}] Patch file written to pod, now attempting to apply patch...")

        # Attempt to apply patch
        exec_command = ["/bin/bash", "-c", """
            cd /testbed && 
            if git apply -v /tmp/patch.diff; then
                echo "APPLY_PATCH_PASS"
            else
                echo "Failed to apply patch with git apply, trying with patch command..."
                if patch --batch --fuzz=5 -p1 -i /tmp/patch.diff; then
                    echo "APPLY_PATCH_PASS"
                else
                    echo "APPLY_PATCH_FAIL"
                fi
            fi
        """]
        logger.info(f"[{instance_id}] Applying patch...")
        output = stream(k8s_api.connect_get_namespaced_pod_exec,
                      name=pod.metadata.name,
                      namespace=K8S_EXECUTOR_NAMESPACE,
                      command=exec_command,
                      stderr=True, stdin=False, stdout=True, tty=False)
        logger.info(f"[{instance_id}] Applying patch output:\n{output}")
        if "APPLY_PATCH_FAIL" in output:
            logger.info(f"[{instance_id}] {APPLY_PATCH_FAIL}:\n{output}")
            raise EvaluationError(
                instance_id,
                f"{APPLY_PATCH_FAIL}:\n{output}",
                logger
            )
        elif "APPLY_PATCH_PASS" in output:
            logger.info(f"[{instance_id}] {APPLY_PATCH_PASS}:\n{output}")
        else:
            logger.info(f"[{instance_id}] Unexpected output when applying patch:\n{output}")
            raise EvaluationError(instance_id, f"Unexpected output when applying patch:\n{output}", logger)

        # Write eval script to pod
        success = write_file_to_k8s_pod(k8s_api, pod.metadata.name, K8S_EXECUTOR_NAMESPACE, 
                                        test_spec.eval_script, "/tmp/eval.sh", logger)
        if not success:
            raise EvaluationError(instance_id, "Failed to write eval script to pod", logger)

        # Make the script executable
        exec_command = ["/bin/bash", "-c", "chmod +x /tmp/eval.sh"]
        output = stream(k8s_api.connect_get_namespaced_pod_exec,
               name=pod.metadata.name,
               namespace=K8S_EXECUTOR_NAMESPACE,
               command=exec_command,
               stderr=True, stdin=False, stdout=True, tty=False)
        logger.info(f"[{instance_id}] Make eval script executable output:\n{output}")

        logger.info(f"[{instance_id}] Eval script written to pod, now attempting to execute...")

        exec_command = "/tmp/eval.sh"
        test_output, timed_out, execution_time = k8s_exec_run_with_timeout(k8s_api, pod.metadata.name, K8S_EXECUTOR_NAMESPACE, exec_command, timeout=timeout)

        if timed_out:
            logger.warning(f"Execution of {instance_id} timed out after {execution_time:.2f} seconds")

        test_output_path = os.path.join(log_dir, "test_output.txt")
        with open(test_output_path, 'w') as f:
            f.write(test_output)
        logger.info(f"Test output for {instance_id} written to {test_output_path}")

        try:
            k8s_api.delete_namespaced_pod(name=pod.metadata.name, namespace=K8S_EXECUTOR_NAMESPACE)
            logger.info(f"Stopped pod {pod.metadata.name}")
        except Exception as e:
            logger.error(f"Error stopping pod {pod.metadata.name}: {str(e)}")

        # Get report from test output
        logger.info(f"Grading answer for {instance_id}...")
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            log_path=test_output_path,
            include_tests_status=True,
        )
        logger.info(f"report: {report}\nResult for {instance_id}: resolved: {report[instance_id]['resolved']}")

        # Write report to report.json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        return instance_id, report

    except EvaluationError as e:
        logger.error(str(e))
        print(e)
    except Exception as e:
        logger.error(f"Error in evaluating model for {instance_id}: {str(e)}")
    finally:
        # Delete pod
        try:
            k8s_api.delete_namespaced_pod(name=pod.metadata.name, namespace=K8S_EXECUTOR_NAMESPACE)
        except:
            pass
        close_logger(logger)
    return

def k8s_run_instances(
    predictions: dict,
    instances: list,
    timeout: int,
    output_dir: str,
    max_workers: int,
):
    """
    Run instances in Kubernetes cluster.

    It will modify "predictions" in-place by adding "instance_id" to each prediction.
    """
    config.load_incluster_config()

    api_client = client.ApiClient(pool_threads=64)
    k8s_api = client.CoreV1Api(api_client=api_client)

    test_specs = list(map(make_test_spec, instances))

    print(f"Running {len(instances)} instances...")
    results = []
    with tqdm(total=len(instances), smoothing=0) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    k8s_run_instance,
                    test_spec,
                    predictions[test_spec.instance_id],
                    k8s_api,
                    timeout,
                    output_dir,
                ): test_spec.instance_id
                for test_spec in test_specs
            }
            # Wait for each future to complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    traceback.print_exc()
                    continue
                finally:
                    print(f"Instance {futures[future]} completed.")
                    pbar.update(1)
    print("All instances run.")
    return results

def k8s_make_run_report(
    predictions: dict,
    full_dataset: list,
    output_dir: str,
) -> None:
    """
    Make a final evaluation and run report of the instances that have been run.
    Also reports on images and containers that may still running!

    Args:
        predictions (dict): Predictions dict generated by the model
        full_dataset (list): List of all instances
        output_dir (str): Path to output directory
    Returns:
        Path to report file
    """
    # instantiate sets to store IDs of different outcomes
    completed_ids = set()
    resolved_ids = set()
    error_ids = set()
    unresolved_ids = set()
    incomplete_ids = set()
    # get instances with empty patches
    empty_patch_ids = set()

    # iterate through dataset and check if the instance has been run
    for instance in full_dataset:
        instance_id = instance[KEY_INSTANCE_ID]
        if instance_id not in predictions:
            # skip instances without 
            incomplete_ids.add(instance_id)
            continue
        prediction = predictions[instance_id]
        if prediction.get("model_patch", None) in ["", None]:
            empty_patch_ids.add(instance_id)
            continue
        report_file = os.path.join(
            output_dir,
            'instances',
            prediction[KEY_INSTANCE_ID],
            "report.json"
        )
        report_file_exists = os.path.exists(report_file)
        if report_file_exists:
            # If report file exists, then the instance has been run
            completed_ids.add(instance_id)
            with open(report_file, 'r') as f:
                report = json.load(f)
            if report[instance_id]["resolved"]:
                # Record if the instance was resolved
                resolved_ids.add(instance_id)
            else:
                unresolved_ids.add(instance_id)
        else:
            # Otherwise, the instance was not run successfully
            error_ids.add(instance_id)


    # print final report
    print(f"Total instances: {len(full_dataset)}")
    print(f"Instances submitted: {len(predictions)}")
    print(f"Instances completed: {len(completed_ids)}")
    print(f"Instances incomplete: {len(incomplete_ids)}")
    print(f"Instances resolved: {len(resolved_ids)}")
    print(f"Instances unresolved: {len(unresolved_ids)}")
    print(f"Instances with empty patches: {len(empty_patch_ids)}")
    print(f"Instances with errors: {len(error_ids)}")

    # write report to file
    report = {
        "total_instances": len(full_dataset),
        "submitted_instances": len(predictions),
        "completed_instances": len(completed_ids),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "empty_patch_instances": len(empty_patch_ids),
        "error_instances": len(error_ids),
        "completed_ids": list(sorted(completed_ids)),
        "incomplete_ids": list(sorted(incomplete_ids)),
        "empty_patch_ids": list(sorted(empty_patch_ids)),
        "submitted_ids": list(sorted(predictions.keys())),
        "resolved_ids": list(sorted(resolved_ids)),
        "unresolved_ids": list(sorted(unresolved_ids)),
        "error_ids": list(sorted(error_ids)),
        "schema_version": 2,
    }
    report_file = os.path.join(
        output_dir,
        "report.json"
    )
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Report written to {report_file}")
    return report_file

def k8s_get_dataset_from_preds(
        dataset_name: str,
        split: str,
        instance_ids: list,
        predictions: dict,
        output_dir: str,
        exclude_completed: bool = True
    ):
    """
    Return only instances that have predictions and are in the dataset.
    If instance_ids is provided, only return instances with those IDs.
    If exclude_completed is True, only return instances that have not been run yet.
    """
    # load dataset
    dataset = load_swebench_dataset(dataset_name, split)
    dataset_ids = {i[KEY_INSTANCE_ID] for i in dataset}

    if instance_ids:
        # check that all instance IDs are in the dataset
        instance_ids = set(instance_ids)
        if instance_ids - dataset_ids:
            raise ValueError(
                (
                    "Some instance IDs not found in dataset!"
                    f"\nMissing IDs:\n{' '.join(instance_ids - dataset_ids)}"
                )
            )
        # check that all instance IDs have predictions
        missing_preds = instance_ids - set(predictions.keys())
        if missing_preds:
            print(f"Warning: Missing predictions for {len(missing_preds)} instance IDs.")
    
    # check that all prediction IDs are in the dataset
    prediction_ids = set(predictions.keys())
    if prediction_ids - dataset_ids:
        raise ValueError(
            (
                "Some prediction IDs not found in dataset!"
                f"\nMissing IDs:\n{' '.join(prediction_ids - dataset_ids)}"
            )
        )

    if instance_ids:
        # filter dataset to just the instance IDs
        dataset = [i for i in dataset if i[KEY_INSTANCE_ID] in instance_ids]

    # check which instance IDs have already been run
    completed_ids = set()
    for instance in dataset:
        if instance[KEY_INSTANCE_ID] not in prediction_ids:
            # skip instances without predictions
            continue
        prediction = predictions[instance[KEY_INSTANCE_ID]]
        report_file = os.path.join(
            output_dir,
            "instances",
            prediction[KEY_INSTANCE_ID],
            "report.json"
        )
        if os.path.exists(report_file):
            completed_ids.add(instance[KEY_INSTANCE_ID])

    if completed_ids and exclude_completed:
        # filter dataset to only instances that have not been run
        print(f"{len(completed_ids)} instances already run, skipping...")
        dataset = [i for i in dataset if i[KEY_INSTANCE_ID] not in completed_ids]

    empty_patch_ids = {k for k, v in predictions.items() if v["model_patch"] == "" or v["model_patch"] is None}

    # filter dataset to only instances with predictions
    dataset = [i for i in dataset if i[KEY_INSTANCE_ID] in prediction_ids and i[KEY_INSTANCE_ID] not in empty_patch_ids]
    return dataset


def k8s_main(
    dataset_name: str,
    split: str,
    instance_ids: list,
    predictions_path: str,
    timeout: int,
    output_dir: str,
    max_workers: int,
):
    # Load predictions
    if predictions_path == 'gold':
        print("Using gold predictions - ignoring predictions_path")
        predictions = get_gold_predictions(dataset_name, split)
    else:
        if predictions_path.endswith(".jsonl.gz"):
            with gzip.open(predictions_path, 'rt') as f:
                predictions = [json.loads(line) for line in f]
        else:
            with open(predictions_path, 'r') as f:
                predictions = json.load(f) if predictions_path.endswith(".json") else [json.loads(line) for line in f]
    predictions = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}

    # Get dataset from predictions
    dataset = k8s_get_dataset_from_preds(dataset_name, split, instance_ids, predictions, output_dir)
    full_dataset = load_swebench_dataset(dataset_name, split)
    print(f"Running {len(dataset)} unevaluated instances...")

    if not dataset:
        print("No instances to run.")
    else:
        # Run instances
        k8s_run_instances(
            predictions,
            dataset,
            timeout,
            output_dir,
            max_workers,
        )
    # Make final report
    client = None  # We don't need Docker client for Kubernetes version
    k8s_make_run_report(predictions, full_dataset, output_dir)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--dataset_name", default="princeton-nlp/SWE-bench_Lite", type=str, help="Name of dataset or path to JSON file.")
    parser.add_argument("--split", type=str, default="test", help="Split of the dataset")
    parser.add_argument("--instance_ids", nargs="+", type=str, help="Instance IDs to run (space separated)")
    parser.add_argument("--predictions_path", type=str, help="Path to predictions file (can be local or in Google Cloud Storage bucket) - if 'gold', uses gold predictions", required=True)
    parser.add_argument("--output_dir", type=str, default="none", help="Path to output directory (can be local or in Google Cloud Storage bucket)")
    parser.add_argument(
        "--timeout", type=int, default=1_800, help="Timeout (in seconds) for running tests for each instance"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=256,
        help="Maximum number of workers to run in parallel"
    )
    args = parser.parse_args()

    gs_output_dir = args.output_dir

    with tempfile.TemporaryDirectory() as local_predictions_dir, tempfile.TemporaryDirectory() as local_output_dir:
        args.output_dir = local_output_dir
        
        # download predictions from GCS
        if args.predictions_path.startswith('gs://'):
            print(f"[init] Downloading predictions from {args.predictions_path} to {local_predictions_dir}")
            storage_client = storage.Client()
            
            # Parse the GCS path for predictions
            pred_bucket_name, pred_blob_name = args.predictions_path[5:].split('/', 1)
            
            # Get the bucket and blob
            bucket = storage_client.bucket(pred_bucket_name)
            blob = bucket.blob(pred_blob_name)
            
            # Download to a local file
            local_pred_file = os.path.join(local_predictions_dir, os.path.basename(pred_blob_name))
            blob.download_to_filename(local_pred_file)
            
            print(f"[init] Predictions downloaded to {local_pred_file}")
            
            # Update the predictions_path to use the local file
            args.predictions_path = local_pred_file

        k8s_main(**vars(args))
        
        # Copy results to GCS
        print(f"[complete] Copying results to {gs_output_dir}")
        from google.cloud import storage

        def upload_local_directory_to_gcs(local_path, bucket_name, gcs_path):
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)

            for local_file in glob.glob(local_path + '/**', recursive=True):
                if os.path.isfile(local_file):
                    remote_path = os.path.join(gcs_path, os.path.relpath(local_file, local_path))
                    blob = bucket.blob(remote_path)
                    blob.upload_from_filename(local_file)
                    print(f"[complete] File {local_file} uploaded to {remote_path}.")

        # Parse the GCS path
        if gs_output_dir.startswith('gs://'):
            bucket_name, gcs_path = gs_output_dir[5:].split('/', 1)
        else:
            raise ValueError("Invalid GCS path. It should start with 'gs://'")

        # Upload the directory
        upload_local_directory_to_gcs(local_output_dir, bucket_name, gcs_path)
        print("Done!")

        # Add `DONE` file to GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path + "/DONE")
        blob.upload_from_string("DONE")
        print("DONE file uploaded to GCS.")
