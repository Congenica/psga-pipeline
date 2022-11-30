from pathlib import Path, PosixPath
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field
from functools import partial, reduce
import json
from json import JSONEncoder

import click
import pandas as pd

from scripts.common.contamination_removal import (
    CONTAMINATION_REMOVAL_SAMPLE_ID_COL,
    EXPECTED_CONTAMINATION_REMOVAL_HEADERS,
)
from scripts.common.primer_autodetection import (
    PRIMER_AUTODETECTION_SAMPLE_ID_COL,
    EXPECTED_PRIMER_AUTODETECTION_HEADERS,
)
from scripts.common.format_genotyping import (
    TYPING_SAMPLE_ID_COL,
    EXPECTED_TYPING_HEADERS,
)
from scripts.util.logger import get_structlog_logger, ERROR, WARNING, INFO
from scripts.util.metadata import EXPECTED_HEADERS as EXPECTED_METADATA_HEADERS, SAMPLE_ID, ILLUMINA, ONT, UNKNOWN
from scripts.util.notifications import Event, Notification
from scripts.util.convert import csv_to_json
from scripts.util.slugs import get_file_with_type, FileType
from scripts.validation.check_csv_columns import check_csv_columns

log_file = f"{Path(__file__).stem}.log"
logger = get_structlog_logger(log_file=log_file)

RESULTFILES_TYPE = Dict[str, List[Dict[str, str]]]

# header for ncov qc summary CSV file
NCOV_SAMPLE_ID_COL = "sample_name"
EXPECTED_NCOV_HEADERS = {
    NCOV_SAMPLE_ID_COL,
    "pct_N_bases",
    "pct_covered_bases",
    "longest_no_N_run",
    "num_aligned_reads",
    "qc_pass",
    "fasta",
    "bam",
}

# header for pangolin lineages CSV file
PANGOLIN_SAMPLE_ID_COL = "taxon"
EXPECTED_PANGOLIN_HEADERS = {
    PANGOLIN_SAMPLE_ID_COL,
    "lineage",
    "conflict",
    "ambiguity_score",
    "scorpio_call",
    "scorpio_support",
    "scorpio_conflict",
    "scorpio_notes",
    "version",
    "pangolin_version",
    "scorpio_version",
    "constellation_version",
    "is_designated",
    "qc_status",
    "qc_notes",
    "note",
}


# These are pipeline generic columns
STATUS = "STATUS"

# these columns point to specific files and are not needed
COLUMNS_TO_REMOVE_FROM_RESULTS_CSV = {
    "FASTA",
    "BAM",
}

# notification event names
UNKNOWN_CONTAMINATION_REMOVAL = "unknown_contamination_removal"
FAILED_CONTAMINATION_REMOVAL = "failed_contamination_removal"
PASSED_CONTAMINATION_REMOVAL = "passed_contamination_removal"
UNKNOWN_PRIMER_AUTODETECTION = "unknown_primer_autodetection"
FAILED_PRIMER_AUTODETECTION = "failed_primer_autodetection"
PASSED_PRIMER_AUTODETECTION = "passed_primer_autodetection"
UNKNOWN_NCOV = "unknown_ncov"
FAILED_NCOV = "failed_ncov"
PASSED_NCOV = "passed_ncov"
UNKNOWN_PANGOLIN = "unknown_pangolin"
FAILED_PANGOLIN = "failed_pangolin"
PASSED_PANGOLIN = "passed_pangolin"


@dataclass
class SampleIdResultFiles:
    """
    Organisation of the expected results files per sample
    """

    # all samples of the analysis run
    all_samples: List[str] = field(metadata={"required": True}, default_factory=list)
    # samples which completed contamination-removal
    contamination_removal_completed_samples: List[str] = field(metadata={"required": True}, default_factory=list)
    # samples which completed primer-autodetection, whether passing or failing primer-autodetection qc
    primer_autodetection_completed_samples: List[str] = field(metadata={"required": True}, default_factory=list)
    # samples which completed ncov, whether passing or failing ncov qc
    ncov_completed_samples: List[str] = field(metadata={"required": True}, default_factory=list)
    # samples passing ncov qc
    ncov_qc_passed_samples: List[str] = field(metadata={"required": True}, default_factory=list)


class PathJSONEncoder(JSONEncoder):
    """
    Enable JSON serialisation of PosixPath objects
    """

    def default(self, o):
        if isinstance(o, PosixPath):
            return str(o)
        return super().default(o)


def load_data_from_csv(
    csv_file: str, expected_columns: Set[str], sample_name_col_to_rename: str = None
) -> pd.DataFrame:
    """
    Load the CSV content to a Pandas dataframe. An arbitrary column name used to indentify the sample id
    can be renamed to "sample_id"
    """
    if csv_file:
        df = pd.read_csv(Path(csv_file))
        check_csv_columns(set(df.columns), expected_columns)
        if sample_name_col_to_rename:
            df = df.rename(columns={sample_name_col_to_rename: SAMPLE_ID})
    else:
        # create a dataframe with header but no rows
        df = pd.DataFrame({c: [] for c in expected_columns})
        df = df.rename(columns={sample_name_col_to_rename: SAMPLE_ID})
    return df


def _generate_contamination_removal_notifications(
    analysis_run_name: str,
    all_samples: List[str],
    df_contamination_removal: pd.DataFrame,
) -> Tuple[List[str], Notification]:
    """
    Generate and publish contamination_removal notifications.
    """
    events = {}

    # initialise contamination_removal as if it had not executed.
    contamination_removal_all_samples = all_samples
    qc_unrelated_failing_contamination_removal_samples = []
    # Currently, there is no QC for contamination removal, so all sample pass
    contamination_removal_samples_failing_qc: List[str] = []
    contamination_removal_samples_passing_qc = all_samples

    if not df_contamination_removal.empty:
        contamination_removal_all_samples = df_contamination_removal[SAMPLE_ID].tolist()
        qc_unrelated_failing_contamination_removal_samples = [
            s for s in all_samples if s not in contamination_removal_all_samples
        ]

        events = {
            UNKNOWN_CONTAMINATION_REMOVAL: Event(
                analysis_run=analysis_run_name,
                level=ERROR,
                message="contamination removal QC unknown",
                samples=qc_unrelated_failing_contamination_removal_samples,
            ),
            FAILED_CONTAMINATION_REMOVAL: Event(
                analysis_run=analysis_run_name,
                level=WARNING,
                message="contamination removal QC failed",
                samples=contamination_removal_samples_failing_qc,
            ),
            PASSED_CONTAMINATION_REMOVAL: Event(
                analysis_run=analysis_run_name,
                level=INFO,
                message="contamination removal QC passed",
                samples=contamination_removal_samples_passing_qc,
            ),
        }

    notifications = Notification(events=events)
    notifications.publish()
    return qc_unrelated_failing_contamination_removal_samples, events


def _generate_primer_autodetection_notifications(
    analysis_run_name: str,
    all_samples: List[str],
    df_primer_autodetection: pd.DataFrame,
) -> Tuple[List[str], Notification]:
    """
    Generate and publish primer_autodetection notifications.
    """
    events = {}

    # initialise primer_autodetection as if it had not executed. This is the default case in which fastas were processed
    primer_autodetection_all_samples = all_samples
    qc_unrelated_failing_primer_autodetection_samples = []
    # Currently, there is no QC for primer autodetection, so all sample pass
    primer_autodetection_samples_failing_qc: List[str] = []
    primer_autodetection_samples_passing_qc = all_samples

    if not df_primer_autodetection.empty:
        primer_autodetection_all_samples = df_primer_autodetection[SAMPLE_ID].tolist()
        qc_unrelated_failing_primer_autodetection_samples = [
            s for s in all_samples if s not in primer_autodetection_all_samples
        ]

        events = {
            UNKNOWN_PRIMER_AUTODETECTION: Event(
                analysis_run=analysis_run_name,
                level=ERROR,
                message="primer autodetection QC unknown",
                samples=qc_unrelated_failing_primer_autodetection_samples,
            ),
            FAILED_PRIMER_AUTODETECTION: Event(
                analysis_run=analysis_run_name,
                level=WARNING,
                message="primer autodetection QC failed",
                samples=primer_autodetection_samples_failing_qc,
            ),
            PASSED_PRIMER_AUTODETECTION: Event(
                analysis_run=analysis_run_name,
                level=INFO,
                message="primer autodetection QC passed",
                samples=primer_autodetection_samples_passing_qc,
            ),
        }

    notifications = Notification(events=events)
    notifications.publish()
    return qc_unrelated_failing_primer_autodetection_samples, events


def _generate_ncov_notifications(
    analysis_run_name: str,
    all_samples: List[str],
    df_ncov_qc: pd.DataFrame,
) -> Tuple[List[str], Notification]:
    """
    Generate and publish ncov notifications.
    Return the list of samples which failed not due to QC
    """
    events = {}

    # initialise ncov as if it had not executed. This is the default case in which fastas were processed
    ncov_all_samples = all_samples
    qc_unrelated_failing_ncov_samples = []
    ncov_samples_passing_qc = all_samples

    if not df_ncov_qc.empty:
        ncov_all_samples = df_ncov_qc[SAMPLE_ID].tolist()
        qc_unrelated_failing_ncov_samples = [s for s in all_samples if s not in ncov_all_samples]
        ncov_samples_failing_qc = df_ncov_qc.loc[~df_ncov_qc["qc_pass"]][SAMPLE_ID]
        ncov_samples_passing_qc = df_ncov_qc.loc[df_ncov_qc["qc_pass"]][SAMPLE_ID]

        events = {
            UNKNOWN_NCOV: Event(
                analysis_run=analysis_run_name,
                level=ERROR,
                message="ncov QC unknown",
                samples=qc_unrelated_failing_ncov_samples,
            ),
            FAILED_NCOV: Event(
                analysis_run=analysis_run_name,
                level=WARNING,
                message="ncov QC failed",
                samples=ncov_samples_failing_qc,
            ),
            PASSED_NCOV: Event(
                analysis_run=analysis_run_name,
                level=INFO,
                message="ncov QC passed",
                samples=ncov_samples_passing_qc,
            ),
        }

    notifications = Notification(events=events)
    notifications.publish()

    return qc_unrelated_failing_ncov_samples, events


def _generate_pangolin_notifications(
    analysis_run_name: str,
    ncov_samples_passing_qc: List[str],
    df_pangolin: pd.DataFrame,
) -> Tuple[List[str], Notification]:
    """
    Generate and publish pangolin notifications.
    Return the list of samples which failed not due to QC
    """
    # pangolin is executed after ncov, unless the latter is skipped (e.g. fasta files)
    pangolin_all_samples = df_pangolin[SAMPLE_ID].tolist()
    # NOTE: pangolin is not executed for samples failing ncov QC
    qc_unrelated_failing_pangolin_samples = [s for s in ncov_samples_passing_qc if s not in pangolin_all_samples]
    pangolin_samples_failing_qc = df_pangolin.loc[df_pangolin["qc_status"] == "fail"][SAMPLE_ID]
    pangolin_samples_passing_qc = df_pangolin.loc[df_pangolin["qc_status"] == "pass"][SAMPLE_ID]

    events = {
        UNKNOWN_PANGOLIN: Event(
            analysis_run=analysis_run_name,
            level=ERROR,
            message="pangolin QC unknown",
            samples=qc_unrelated_failing_pangolin_samples,
        ),
        FAILED_PANGOLIN: Event(
            analysis_run=analysis_run_name,
            level=WARNING,
            message="pangolin QC failed",
            samples=pangolin_samples_failing_qc,
        ),
        PASSED_PANGOLIN: Event(
            analysis_run=analysis_run_name,
            level=INFO,
            message="pangolin QC passed",
            samples=pangolin_samples_passing_qc,
        ),
    }

    notifications = Notification(events=events)
    notifications.publish()

    return qc_unrelated_failing_pangolin_samples, events


def _generate_notifications(
    analysis_run_name: str,
    all_samples: List[str],
    df_contamination_removal: pd.DataFrame,
    df_primer_autodetection: pd.DataFrame,
    df_ncov_qc: pd.DataFrame,
    df_pangolin: pd.DataFrame,
) -> Tuple[List[str], Notification]:
    """
    Generate and publish output pipeline notifications.
    Return the list of samples which failed not due to QC
    """
    (
        qc_unrelated_failing_contamination_removal_samples,
        contamination_removal_events,
    ) = _generate_contamination_removal_notifications(
        analysis_run_name,
        all_samples,
        df_contamination_removal,
    )

    contamination_removal_processed_samples = (
        contamination_removal_events[PASSED_CONTAMINATION_REMOVAL].samples if contamination_removal_events else []
    )

    (
        qc_unrelated_failing_primer_autodetection_samples,
        primer_autodetection_events,
    ) = _generate_primer_autodetection_notifications(
        analysis_run_name,
        contamination_removal_processed_samples,
        df_primer_autodetection,
    )

    primer_autodetection_processed_samples = (
        primer_autodetection_events[PASSED_PRIMER_AUTODETECTION].samples if primer_autodetection_events else []
    )
    qc_unrelated_failing_ncov_samples, ncov_events = _generate_ncov_notifications(
        analysis_run_name,
        primer_autodetection_processed_samples,
        df_ncov_qc,
    )

    if contamination_removal_events:
        if ncov_events:
            pangolin_processed_samples = ncov_events[PASSED_NCOV].samples
        else:
            pangolin_processed_samples = []
    else:
        # fasta files
        pangolin_processed_samples = all_samples

    qc_unrelated_failing_pangolin_samples, pangolin_events = _generate_pangolin_notifications(
        analysis_run_name,
        pangolin_processed_samples,
        df_pangolin,
    )
    events = {
        **contamination_removal_events,
        **primer_autodetection_events,
        **ncov_events,
        **pangolin_events,
    }
    qc_unrelated_failing_samples = list(
        set(qc_unrelated_failing_contamination_removal_samples)
        | set(qc_unrelated_failing_primer_autodetection_samples)
        | set(qc_unrelated_failing_ncov_samples)
        | set(qc_unrelated_failing_pangolin_samples)
    )

    return qc_unrelated_failing_samples, events


def _generate_results_csv(
    all_samples: List[str],
    df_contamination_removal: pd.DataFrame,
    df_primer_autodetection: pd.DataFrame,
    df_ncov_qc: pd.DataFrame,
    df_ncov_typing: pd.DataFrame,
    df_pangolin: pd.DataFrame,
    qc_unrelated_failing_samples: List[str],
    output_results_csv_file: str,
) -> None:
    """
    Generate the pipeline results CSV file.
    """

    status_col_data = [
        {SAMPLE_ID: s, STATUS: "Failed"} if s in qc_unrelated_failing_samples else {SAMPLE_ID: s, STATUS: "Completed"}
        for s in all_samples
    ]
    df_status = pd.DataFrame(status_col_data)

    # list of dataframes to merge. They share 1 shared column: SAMPLE_ID
    dfs_to_merge = [
        df_status,
        df_contamination_removal,
        df_primer_autodetection,
        df_ncov_qc,
        df_ncov_typing,
        df_pangolin,
    ]

    # partial stores part of a function’s arguments resulting in a new object with a simplified signature.
    # reduce applies cumulatively the new partial object to the items of iterable (list of dataframes here).
    merge = partial(pd.merge, how="outer")
    df_merged = reduce(merge, dfs_to_merge)

    # upper case column header
    df_merged.columns = [col.upper() for col in df_merged.columns]
    # move sample_id col to first column
    sample_id_col_data = df_merged.pop(SAMPLE_ID)
    df_merged.insert(0, SAMPLE_ID, sample_id_col_data)

    # remove unwanted columns
    df_merged.drop(COLUMNS_TO_REMOVE_FROM_RESULTS_CSV, axis=1, inplace=True)
    # save to CSV
    df_merged.to_csv(output_results_csv_file, encoding="utf-8", index=False)


def get_expected_output_files_per_sample(
    output_path: str,
    sample_ids_result_files: SampleIdResultFiles,
    sequencing_technology: str,
) -> RESULTFILES_TYPE:
    """
    Return a dictionary {sample_id, list_of_expected_output_paths}
    """
    # initialise the dictionary keys
    output_files: RESULTFILES_TYPE = {sample_id: [] for sample_id in sample_ids_result_files.all_samples}

    def _append_reheadered_fasta(output_path, sample_id, output_files):
        output_files[sample_id].extend(
            get_file_with_type(
                output_path=output_path,
                inner_dirs=["reheadered-fasta"],
                filetypes=[FileType(".fasta", "fasta/final")],
                sample_id=sample_id,
            )
        )

    if sequencing_technology == UNKNOWN:
        # FASTA samples
        for sample_id in sample_ids_result_files.all_samples:
            # expected files for all samples
            _append_reheadered_fasta(output_path, sample_id, output_files)
    else:
        if sequencing_technology == ILLUMINA:
            contamination_removal_clean_fastq = [
                FileType(f"_{r}.fastq.gz", "fastq/cleaned-sequence-data") for r in [1, 2]
            ]
            fastqc = [FileType(f"_{r}_fastqc.zip", "fastqc/qc") for r in [1, 2]]
            ncov_bam = [
                FileType(".mapped.primertrimmed.sorted.bam", "bam/final"),
                FileType(".mapped.primertrimmed.sorted.bam.bai", "bai/final"),
            ]
            ncov_fasta = [FileType(".primertrimmed.consensus.fa", "fasta/consensus")]
            ncov_variants = [FileType(".variants.tsv", "tsv/final")]
        elif sequencing_technology == ONT:
            contamination_removal_clean_fastq = [FileType(".fastq.gz", "fastq/cleaned-sequence-data")]
            fastqc = [FileType("_fastqc.zip", "fastqc/qc")]
            ncov_bam = [
                FileType(".primertrimmed.rg.sorted.bam", "bam/final"),
                FileType(".primertrimmed.rg.sorted.bam.bai", "bai/final"),
            ]
            ncov_fasta = [
                FileType(".consensus.fa", "fasta/consensus"),
                FileType(".preconsensus.fa", "fasta/preconsensus"),
            ]
            ncov_variants = [
                FileType(".pass.vcf.gz", "vcf/final"),
                FileType(".pass.vcf.gz.tbi", "tbi/final"),
            ]
        else:
            raise ValueError(f"Unsupported sequencing_technology: {sequencing_technology}")

        ncov_typing = [
            FileType(".typing.csv", "csv/typing"),
            FileType(".variants.csv", "csv/typing-variants"),
            FileType(".csq.vcf", "vcf/typing-consequences"),
        ]
        primer_autodetection_csvs = [
            FileType("_primer_data.csv", "csv/primer-data"),
            FileType("_primer_detection.csv", "csv/primer-detection"),
        ]

        for sample_id in sample_ids_result_files.all_samples:
            # expected files for all samples
            output_files[sample_id].extend(
                get_file_with_type(
                    output_path=output_path,
                    inner_dirs=["contamination_removal", "cleaned_fastq"],
                    filetypes=contamination_removal_clean_fastq,
                    sample_id=sample_id,
                )
            )
            output_files[sample_id].extend(
                get_file_with_type(
                    output_path=output_path,
                    inner_dirs=["contamination_removal", "counting"],
                    filetypes=[FileType(".txt", "txt/cleaned-sequence-count")],
                    sample_id=sample_id,
                )
            )
            output_files[sample_id].extend(
                get_file_with_type(
                    output_path=output_path,
                    inner_dirs=["contamination_removal"],
                    filetypes=[FileType("_contamination_removal.csv", "csv/cleaned-sequence-count")],
                    sample_id=sample_id,
                )
            )
            output_files[sample_id].extend(
                get_file_with_type(
                    output_path=output_path,
                    inner_dirs=["fastqc"],
                    filetypes=fastqc,
                    sample_id=sample_id,
                )
            )

        for sample_id in sample_ids_result_files.primer_autodetection_completed_samples:
            output_files[sample_id].extend(
                get_file_with_type(
                    output_path=output_path,
                    inner_dirs=["primer_autodetection"],
                    filetypes=primer_autodetection_csvs,
                    sample_id=sample_id,
                )
            )

        for sample_id in sample_ids_result_files.ncov_completed_samples:
            # expected files for samples which completed ncov
            output_files[sample_id].extend(
                get_file_with_type(
                    output_path=output_path,
                    inner_dirs=["ncov2019-artic", "output_bam"],
                    filetypes=ncov_bam,
                    sample_id=sample_id,
                )
            )
            output_files[sample_id].extend(
                get_file_with_type(
                    output_path=output_path,
                    inner_dirs=["ncov2019-artic", "output_fasta"],
                    filetypes=ncov_fasta,
                    sample_id=sample_id,
                )
            )
            output_files[sample_id].extend(
                get_file_with_type(
                    output_path=output_path,
                    inner_dirs=["ncov2019-artic", "output_typing"],
                    filetypes=ncov_typing,
                    sample_id=sample_id,
                )
            )
            output_files[sample_id].extend(
                get_file_with_type(
                    output_path=output_path,
                    inner_dirs=["ncov2019-artic", "output_variants"],
                    filetypes=ncov_variants,
                    sample_id=sample_id,
                )
            )
            output_files[sample_id].extend(
                get_file_with_type(
                    output_path=output_path,
                    inner_dirs=["ncov2019-artic", "output_plots"],
                    filetypes=[FileType(".depth.png", "png/qc")],
                    sample_id=sample_id,
                )
            )

        for sample_id in sample_ids_result_files.ncov_qc_passed_samples:
            # expected files for samples which passed ncov QC
            _append_reheadered_fasta(output_path, sample_id, output_files)

    return output_files


def _generate_resultfiles_json(
    sequencing_technology: str,
    sample_ids_result_files: SampleIdResultFiles,
    output_path: str,
    output_resultfiles_json_file: Path,
) -> None:
    """
    Generate a JSON file containing the list of expected result files per sample
    """
    output_files_per_sample = get_expected_output_files_per_sample(
        output_path=output_path,
        sample_ids_result_files=sample_ids_result_files,
        sequencing_technology=sequencing_technology,
    )

    with open(output_resultfiles_json_file, "w") as outfile:
        json.dump(output_files_per_sample, outfile, cls=PathJSONEncoder, sort_keys=True, indent=4)


@click.command()
@click.option("--analysis-run-name", required=True, type=str, help="The name of the analysis run")
@click.option(
    "--metadata-file",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
    help="the sample metadata file",
)
@click.option(
    "--contamination-removal-csv-file",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="contamination removal pipeline resulting csv file",
)
@click.option(
    "--primer-autodetection-csv-file",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="primer autodetection pipeline resulting csv file",
)
@click.option(
    "--ncov-qc-csv-file",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="ncov pipeline qc csv file",
)
@click.option(
    "--ncov-typing-csv-file",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="ncov pipeline typing csv file",
)
@click.option(
    "--pangolin-csv-file",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
    help="pangolin pipeline resulting csv file",
)
@click.option(
    "--output-results-csv-file",
    type=click.Path(dir_okay=False, writable=True),
    default="results.csv",
    help="CSV file containing the summary of results",
)
@click.option(
    "--output-results-json-file",
    type=click.Path(dir_okay=False, writable=True),
    default="results.json",
    help="JSON file containing the summary of results",
)
@click.option(
    "--output-resultfiles-json-file",
    type=click.Path(dir_okay=False, writable=True),
    default="resultfiles.json",
    help="JSON file containing all the expected files per sample",
)
@click.option(
    "--output-path",
    type=str,
    required=True,
    help="output_path output path where sample result files are stored (e.g. s3://bucket/path/analysis_run)",
)
@click.option(
    "--sequencing-technology",
    type=click.Choice([ILLUMINA, ONT, UNKNOWN], case_sensitive=True),
    required=True,
    help="the sequencer technology used for sequencing the samples",
)
def generate_pipeline_results_files(
    analysis_run_name: str,
    metadata_file: str,
    contamination_removal_csv_file: str,
    primer_autodetection_csv_file: str,
    ncov_qc_csv_file: str,
    ncov_typing_csv_file: str,
    pangolin_csv_file: str,
    output_results_csv_file: str,
    output_results_json_file: str,
    output_resultfiles_json_file: str,
    output_path: str,
    sequencing_technology: str,
) -> None:
    """
    Generate pipeline results files
    """
    # data loading
    df_metadata = load_data_from_csv(metadata_file, EXPECTED_METADATA_HEADERS)
    df_contamination_removal = load_data_from_csv(
        contamination_removal_csv_file,
        EXPECTED_CONTAMINATION_REMOVAL_HEADERS,
        CONTAMINATION_REMOVAL_SAMPLE_ID_COL,
    )
    df_primer_autodetection = load_data_from_csv(
        primer_autodetection_csv_file,
        EXPECTED_PRIMER_AUTODETECTION_HEADERS,
        PRIMER_AUTODETECTION_SAMPLE_ID_COL,
    )
    df_ncov_qc = load_data_from_csv(ncov_qc_csv_file, EXPECTED_NCOV_HEADERS, NCOV_SAMPLE_ID_COL)
    df_ncov_typing = load_data_from_csv(ncov_typing_csv_file, EXPECTED_TYPING_HEADERS, TYPING_SAMPLE_ID_COL)
    df_pangolin = load_data_from_csv(pangolin_csv_file, EXPECTED_PANGOLIN_HEADERS, PANGOLIN_SAMPLE_ID_COL)

    all_samples = df_metadata[SAMPLE_ID].tolist()
    # this simplifies testing and result inspection
    all_samples.sort()

    # typing runs as part of ncov, so they share the notifications
    qc_unrelated_failing_samples, events = _generate_notifications(
        analysis_run_name, all_samples, df_contamination_removal, df_primer_autodetection, df_ncov_qc, df_pangolin
    )

    _generate_results_csv(
        all_samples,
        df_contamination_removal,
        df_primer_autodetection,
        df_ncov_qc,
        df_ncov_typing,
        df_pangolin,
        qc_unrelated_failing_samples,
        output_results_csv_file,
    )

    csv_to_json(output_results_csv_file, output_results_json_file, SAMPLE_ID)

    sample_ids_result_files = SampleIdResultFiles(
        all_samples=all_samples,
        contamination_removal_completed_samples=[]
        if sequencing_technology == UNKNOWN
        or not {FAILED_CONTAMINATION_REMOVAL, PASSED_CONTAMINATION_REMOVAL} <= events.keys()
        else list(
            set(events[FAILED_CONTAMINATION_REMOVAL].samples) | set(events[PASSED_CONTAMINATION_REMOVAL].samples)
        ),
        primer_autodetection_completed_samples=[]
        if sequencing_technology == UNKNOWN
        or not {FAILED_PRIMER_AUTODETECTION, PASSED_PRIMER_AUTODETECTION} <= events.keys()
        else list(set(events[FAILED_PRIMER_AUTODETECTION].samples) | set(events[PASSED_PRIMER_AUTODETECTION].samples)),
        ncov_completed_samples=[]
        if sequencing_technology == UNKNOWN or not {FAILED_NCOV, PASSED_NCOV} <= events.keys()
        else list(set(events[FAILED_NCOV].samples) | set(events[PASSED_NCOV].samples)),
        ncov_qc_passed_samples=[]
        if sequencing_technology == UNKNOWN or PASSED_NCOV not in events
        else list(events[PASSED_NCOV].samples),
    )

    _generate_resultfiles_json(
        sequencing_technology, sample_ids_result_files, output_path, Path(output_resultfiles_json_file)
    )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    generate_pipeline_results_files()
