import json
from os.path import join as join_path  # used to join FS paths and S3 URIs

from app.scripts.generate_pipeline_results_files import get_expected_output_files_per_sample, SampleIdResultFiles
from app.scripts.util.metadata import UNKNOWN


def get_expected_output_files(output_path: str, sample_ids: list[str], sequencing_technology: str) -> list[str]:
    """
    Return a list of ALL expected output paths
    """

    # In integration tests, we expect all samples to pass ncov qc if this is executed
    sample_ids_result_files = SampleIdResultFiles(
        all_samples=sample_ids,
        primer_autodetection_completed_samples=[] if sequencing_technology == UNKNOWN else sample_ids,
        ncov_completed_samples=[] if sequencing_technology == UNKNOWN else sample_ids,
    )

    output_files_per_sample = get_expected_output_files_per_sample(
        output_path=str(output_path),
        sample_ids_result_files=sample_ids_result_files,
        sequencing_technology=sequencing_technology,
    )
    # generate a unified list of paths as non-sample results files must also be included
    output_files = [f["file"] for sample_files in output_files_per_sample.values() for f in sample_files]

    output_files.extend([join_path(output_path, "logs", f) for f in ["generate_pipeline_results_files.log"]])

    if sequencing_technology != UNKNOWN:
        output_files.append(join_path(output_path, "ncov2019-artic", "ncov_qc.csv"))

    output_files.append(join_path(output_path, "pangolin", "all_lineages_report.csv"))
    output_files.append(join_path(output_path, "results.csv"))
    output_files.append(join_path(output_path, "resultfiles.json"))

    return output_files


def get_expected_output_files_from_result_files(output_path: str, sequencing_technology: str) -> list[str]:
    """
    Return a list of ALL expected output paths
    """

    # Use the result files instead of reimplementing this logic
    output_files_per_sample = {}
    with open(join_path(output_path, "resultfiles.json")) as resultfiles:
        output_files_per_sample = json.loads(resultfiles.read())

    # generate a unified list of paths as non-sample results files must also be included
    output_files = [f["file"] for sample_files in output_files_per_sample.values() for f in sample_files]

    output_files.extend([join_path(output_path, "logs", f) for f in ["generate_pipeline_results_files.log"]])

    if sequencing_technology != UNKNOWN:
        output_files.append(join_path(output_path, "ncov2019-artic", "ncov_qc.csv"))

    output_files.append(join_path(output_path, "pangolin", "all_lineages_report.csv"))
    output_files.append(join_path(output_path, "results.csv"))
    output_files.append(join_path(output_path, "resultfiles.json"))

    return output_files
