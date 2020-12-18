#!/usr/bin/env python

import csv
from datetime import date
import re

import click
from click import ClickException
from sqlalchemy import String, cast, func

from scripts.db.database import session_handler
from scripts.db.models import Area, Governorate, Sample

EXPECTED_HEADERS = [
    "MRN",
    "AGE",
    "NATIONALITY",
    "GOVERNORATE",
    "AREA",
    "BLOCK",
    "SAMPLE ID",
    "ASSIGN DATE",
    "CT",
    "SYMPTOMS",
]


def _validate_and_normalise_row(session, row):
    # strip leading and trailing spaces from everything
    for f in EXPECTED_HEADERS:
        row[f] = row[f].lstrip().rstrip() if row[f] is not None else ""

    errs = []

    # MRN contains digits. If the person is a foreigner, a 'T' is prefixed
    if not re.match(r"(T)?\d+$", row["MRN"]):
        errs.append(f"MRN \"{row['MRN']}\" is expected to be an integer, prefixed by 'T' if the person is a foreigner")

    # AGE can come in two different formats:
    # AGE can be digits, space, "Y". turn into just digits. if not Y make it 0
    # AGE can be digits-only
    age_column_value = row["AGE"]
    age_y = re.match(r"(\d{1,3}) ([A-Za-z])$", age_column_value)
    if age_y:
        if age_y.group(2).lower() == "y":
            row["AGE"] = int(age_y.group(1))
        else:
            row["AGE"] = 0
    else:
        age_digit = re.match(r"(\d{1,3})$", age_column_value)
        if age_digit:
            row["AGE"] = int(age_digit.group(1))
        else:
            errs.append(
                f"AGE \"{row['AGE']}\" does not follow any known age formats. It needs to be a number-only "
                f"or a number followed by a space then a letter (probably Y)"
            )

    # NATIONALITY should be str
    if not re.match(r"[\w ]+$", row["NATIONALITY"]):
        errs.append(f"NATIONALITY \"{row['NATIONALITY']}\" doesn't look like words")

    # GOVERNORATE should be in governorate enum
    governorate = re.match(r"([\w ]+)$", row["GOVERNORATE"])
    if governorate:
        found_governorate = (
            session.query(Governorate)
            .filter(func.lower(cast(Governorate.name, String)) == governorate.group(1).lower())
            .one_or_none()
        )
        if found_governorate:
            row["GOVERNORATE"] = found_governorate
        else:
            errs.append(f"GOVERNORATE \"{row['GOVERNORATE']}\" looked right, but not found in database")
    elif row["GOVERNORATE"]:
        errs.append(f"GOVERNORATE \"{row['GOVERNORATE']}\" doesn't look like words")

    # AREA should be in the area table
    area = re.match(r"([\w /']+)$", row["AREA"])
    if area:
        # do a case insensitive match just in case, also the spaces around punctuation is inconsistent, so we compare
        # ignoring whitespace
        whitespace = re.compile(r"\s+")
        normalised_area = whitespace.sub("", area.group(1).lower())
        found_area = (
            session.query(Area)
            .filter(func.regexp_replace(func.lower(Area.name), r"\s+", "", "g") == normalised_area)
            .one_or_none()
        )
        if found_area:
            row["AREA"] = found_area
        else:
            errs.append(f"AREA \"{row['AREA']}\" looked right, but not found in database")
    elif row["AREA"]:
        errs.append(f"AREA \"{row['AREA']}\" doesn't look like words")

    # BLOCK should be a number, appears to normally be 3 digits, but have also seen 4, so we'll be lax on this
    if row["BLOCK"] and not re.match(r"(\d+)$", row["BLOCK"]):
        errs.append(f"BLOCK \"{row['BLOCK']}\" should contain digits")

    # SAMPLE ID should be numbers, but not an integer
    if not re.match(r"\d+$", row["SAMPLE ID"]):
        errs.append(f"SAMPLE ID \"{row['SAMPLE ID']}\" is not a number")

    # ASSIGN DATE should be dd/mm/yyyy
    assign_date_match = re.match(r"(\d{2})/(\d{2})/(\d{4})$", row["ASSIGN DATE"])
    if assign_date_match:
        try:
            row["ASSIGN DATE"] = date(
                int(assign_date_match.group(3)), int(assign_date_match.group(2)), int(assign_date_match.group(1))
            )
        except ValueError as e:
            errs.append(f"ASSIGN DATE \"{row['ASSIGN DATE']}\" is not a valid date: {e}")
    else:
        errs.append(f"ASSIGN DATE \"{row['ASSIGN DATE']}\" doesn't look like a date")

    # CT should be CT: followed by a float, which might not have the fractional part
    # it's also not in every row in the example data, so assuming it's optional
    ct = re.match(r"CT:(\d+(?:\.\d+)?)$", row["CT"])
    if ct:
        row["CT"] = float(ct.group(1))
    else:
        row["CT"] = None

    # SYMPTOMS doesn't really need validating

    if errs:
        raise ValueError("\n".join(errs))

    return row


@click.command()
@click.option("--file", required=True, type=click.File("r"), help="The metadata TSV input file")
@click.option(
    "--output_file_for_samples_with_metadata",
    required=False,
    type=click.Path(file_okay=True, writable=True),
    help="File path to populate with all sample names, which have metadata loaded in the database",
)
def load_iseha_data(file, output_file_for_samples_with_metadata):
    """
    Read in a TSV file of I-SEHA metadata and load each row into the database. Invalid rows are warned about, but
    skipped over.

    :param File file: the TSV file to load
    """
    reader = csv.DictReader(file, delimiter="\t")

    headers = reader.fieldnames
    if set(headers) != set(EXPECTED_HEADERS):
        err = "Unexpected TSV headers, got:\n" + ", ".join(headers) + "\nexpected\n" + ", ".join(EXPECTED_HEADERS)
        raise ClickException(err)

    inserted = set()
    updated = set()
    errors = set()
    with session_handler() as session:
        for row in reader:
            try:
                row = _validate_and_normalise_row(session, row)
                existing_sample = session.query(Sample).filter(Sample.lab_id == row["SAMPLE ID"]).one_or_none()
                governorate = row["GOVERNORATE"] if row["GOVERNORATE"] else None
                area = row["AREA"] if row["AREA"] else None
                block = row["BLOCK"] if row["BLOCK"] else None
                if existing_sample:
                    existing_sample.mrn = row["MRN"]
                    existing_sample.age = row["AGE"]
                    existing_sample.nationality = row["NATIONALITY"]
                    existing_sample.governorate = governorate
                    existing_sample.area = area
                    existing_sample.block_number = block
                    existing_sample.date_collected = row["ASSIGN DATE"]
                    existing_sample.ct_value = row["CT"]
                    existing_sample.symptoms = row["SYMPTOMS"]
                    existing_sample.metadata_loaded = True
                    updated.add(row["SAMPLE ID"])
                else:
                    sample = Sample(
                        mrn=row["MRN"],
                        age=row["AGE"],
                        nationality=row["NATIONALITY"],
                        governorate=governorate,
                        area=area,
                        block_number=block,
                        lab_id=row["SAMPLE ID"],
                        sample_number=int(row["SAMPLE ID"]),
                        date_collected=row["ASSIGN DATE"],
                        ct_value=row["CT"],
                        symptoms=row["SYMPTOMS"],
                        metadata_loaded=True,
                    )
                    session.add(sample)
                    inserted.add(row["SAMPLE ID"])
            except ValueError as e:
                click.echo(f"Invalid row for sample ID {row['SAMPLE ID']}:\n{e}", err=True)
                errors.add(row["SAMPLE ID"])

    if errors:
        raise ClickException("Errors encountered: " + ", ".join(map(str, errors)))
    click.echo("Inserted samples: " + ", ".join(map(str, inserted)))
    click.echo("Updated samples: " + ", ".join(map(str, updated)))

    if output_file_for_samples_with_metadata:
        with open(output_file_for_samples_with_metadata, "w") as f:
            with session_handler() as session:
                all_samples_with_metadata = session.query(Sample).filter(Sample.metadata_loaded).all()
                sample_names = [x.lab_id for x in all_samples_with_metadata]
                for sample in sample_names:
                    f.write(f"{sample}\n")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    load_iseha_data()
