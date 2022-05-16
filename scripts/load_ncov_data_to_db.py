import os
import csv
from pathlib import Path
from typing import Dict, List
import click
from click import ClickException
from sqlalchemy.orm import scoped_session, joinedload

from scripts.db.database import session_handler
from scripts.db.models import AnalysisRun, Sample, SampleQC
from scripts.util.data_dumping import write_list_to_file


def load_ncov_sample(
    session: scoped_session, analysis_run_name: str, sample_from_csv: Dict, ncov_qc_depth_directory: str
):
    """
    Load the ncov results of a sample to the database
    """
    sample_name = sample_from_csv["sample_name"]
    sample_qc_depth_file_path = os.path.join(ncov_qc_depth_directory, f"{sample_name}.depth.png")
    if not os.path.isfile(sample_qc_depth_file_path):
        raise ValueError(f"File {sample_qc_depth_file_path}, required to submit sample {sample_name}, does not exist")

    sample = (
        session.query(Sample)
        .join(AnalysisRun)
        .filter(
            Sample.sample_name == sample_name,
            AnalysisRun.analysis_run_name == analysis_run_name,
        )
        .options(joinedload(Sample.sample_qc))
        .one_or_none()
    )

    if not sample:
        raise ClickException(f"Sample name: {sample_name} was not found")

    if not sample.sample_qc:
        sample.sample_qc = SampleQC()

    sample_qc = sample.sample_qc
    sample_qc.pct_n_bases = sample_from_csv["pct_N_bases"]
    sample_qc.pct_covered_bases = sample_from_csv["pct_covered_bases"]
    sample_qc.longest_no_n_run = sample_from_csv["longest_no_N_run"]
    sample_qc.num_aligned_reads = sample_from_csv["num_aligned_reads"]
    sample_qc.qc_pass = sample_from_csv["qc_pass"].lower() == "true"
    with open(sample_qc_depth_file_path, "rb") as f:
        sample_qc.qc_plot = bytearray(f.read())


def get_samples_without_ncov_qc(session: scoped_session, analysis_run_name: str) -> List[str]:
    """
    Return the list of samples without ncov QC record. For these samples, ncov has not executed
    as expected (e.g. ncov failed to run).
    """
    samples = (
        session.query(Sample)
        .join(AnalysisRun)
        .filter(AnalysisRun.analysis_run_name == analysis_run_name)
        .options(joinedload(Sample.sample_qc))
        .all()
    )

    samples_without_qc = [s.sample_name for s in samples if not s.sample_qc]

    return samples_without_qc


@click.command()
@click.option(
    "--ncov-qc-csv-file",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
    help="ncov pipeline resulting qc csv file",
)
@click.option(
    "--ncov-qc-depth-directory",
    type=click.Path(exists=True, dir_okay=True, readable=True),
    required=True,
    help="directory, containing qc depth files, following the pattern {sample_name}.depth.png",
)
@click.option(
    "--samples-without-qc-file",
    type=click.Path(dir_okay=False, writable=True),
    required=True,
    help="output file storing the names of samples for this analysis run which do not have any sample_qc record",
)
@click.option("--analysis-run-name", required=True, type=str, help="The name of the analysis run")
def load_ncov_data(
    ncov_qc_csv_file: str, ncov_qc_depth_directory: str, samples_without_qc_file: str, analysis_run_name: str
) -> None:
    """
    Submit samples QC to the database, generated by ncov pipeline
    """
    with session_handler() as session:
        with open(ncov_qc_csv_file) as csv_file:
            sample_from_csv_reader = csv.DictReader(csv_file)
            for sample_from_csv in sample_from_csv_reader:
                load_ncov_sample(session, analysis_run_name, sample_from_csv, ncov_qc_depth_directory)

        samples_without_qc = get_samples_without_ncov_qc(session, analysis_run_name)
        write_list_to_file(samples_without_qc, Path(samples_without_qc_file))


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    load_ncov_data()
