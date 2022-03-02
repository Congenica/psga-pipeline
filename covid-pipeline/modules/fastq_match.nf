include { makeFastqSearchPath } from './utils.nf'

include { concat_elements_to_single_string as concat_metadata_samples } from './utils.nf'
include { concat_elements_to_single_string as concat_fastq_samples } from './utils.nf'

include { append_match_to_values_list as append_match_to_current_session_samples }  from './utils.nf'
include { append_match_to_values_list as append_qc_pass_match_to_fastq_samples }  from './utils.nf'

include { store_notification_with_values_list as store_notification_no_fastq_file } from './utils.nf'
include { store_notification_with_values_list as store_notification_updated } from './utils.nf'
include { store_notification_with_values_list as store_notification_fastq_only } from './utils.nf'
include { store_notification_with_values_list as store_notification_processed_already } from './utils.nf'


workflow filter_fastq_matching_with_metadata{
    take:
        ch_all_samples_with_metadata_loaded
        ch_current_session_samples_with_metadata_loaded
        ch_qc_passed_samples
        ch_updated_samples
    main:
        // Original usage of grouping illumina fastq file pairs and extracting sample name
        ch_fastq_search_path = makeFastqSearchPath(
            params.illuminaPrefixes,
            params.illuminaSuffixes,
            params.fastq_exts
        )
        Channel.fromFilePairs( ch_fastq_search_path, flat: true)
            .filter{ !( it[0] =~ /Undetermined/ ) }
            .set{ ch_fastq_file_pairs }
        // ch_fastq_file_pairs: [[sample_name, fastq1, fastq2], ...]

        ch_fastq_file_pairs
            .map { it ->  it[0] }
            .flatten()
            .set{ ch_fastq_sample_names }

        // Matching our extracted sample names with samples loaded from the metadata
        ch_fastq_file_pair_matches = append_metadata_match_to_sample_file_pair(
            ch_fastq_file_pairs,
            ch_all_samples_with_metadata_loaded.collect()
        )

        ch_fastq_file_pair_matches
            .branch { it ->
                matched: it[3]
                    return it
                mismatched: true
                    return it
            }
            .set{ ch_fastq_file_pairs_grouped_by_metadata_search }

        ch_fastq_file_pairs_grouped_by_metadata_search.matched
             .map { it -> [ it[1], it[2] ] }
             .flatten()
             .set{ ch_fasta_matching_metadata }

        ch_fastq_file_pairs_grouped_by_metadata_search.mismatched
             .map { it -> [ it[1], it[2] ] }
             .flatten()
             .set{ ch_fasta_mismatching_metadata }
        ch_fastq_file_pairs_grouped_by_metadata_search.mismatched
             .map { it -> it[0] }
             .flatten()
             .set{ ch_mismatching_metadata_sample_names }

        store_fastas_not_matched(ch_fasta_mismatching_metadata)

        ch_current_session_metadata_sample_matches = append_match_to_current_session_samples(
            ch_current_session_samples_with_metadata_loaded,
            ch_fastq_sample_names.collect()
        )

        ch_current_session_metadata_sample_matches
            .branch { it ->
                matched: it[1]
                    return it[0]
                mismatched: true
                    return it[0]
            }
            .set{ ch_current_session_metadata_sample_matches_search }

        ch_fastq_sample_qc_pass_matches = append_qc_pass_match_to_fastq_samples(
            ch_fastq_sample_names,
            ch_qc_passed_samples.collect()
        )

        ch_fastq_sample_qc_pass_matches
            .branch { it ->
                matched: it[1]
                    return it[0]
                mismatched: true
                    return it[0]
            }
            .set{ ch_fastq_sample_qc_pass_matches_search }

        ch_current_session_metadata_sample_matches_search.mismatched.map {
            it -> log.warn """Sample ${it}, read from metadata file, has no matching fastq file in current session."""
        }
        store_notification_no_fastq_file(
          "${workflow.start}-metadata_samples_missing_files.txt",
          ch_current_session_metadata_sample_matches_search.mismatched.collect()
        ).subscribe onNext: {
            log.error("ERROR: Found metadata samples with missing files. See: metadata_samples_missing_files.txt. Aborting!")
            System.exit(1)
        }

        ch_updated_samples.map {
            it -> log.info """Sample ${it} was updated with provided metadata"""
        }
        store_notification_updated(
          "${workflow.start}-updated_samples.txt",
          ch_updated_samples.collect()
        )

        ch_fastq_file_pairs_grouped_by_metadata_search.mismatched.map {
            it -> log.warn """Sample ${it[0]} was not found in database with metadata.
                The following files will not be processed:
                  - ${it[1]}
                  - ${it[2]}
            """
        }
        store_notification_fastq_only(
          "${workflow.start}-no_metadata_provided_for_fastq_samples.txt",
          ch_mismatching_metadata_sample_names.collect()
        )


        ch_fastq_sample_qc_pass_matches_search.matched.map {
          it -> log.warn """Sample ${it} was provided with fastq files. It was processed by pipeline before and was marked as QC_PASS by artic-ncov2019.
              Sample will be processed by pipeline again and will overwrite previous results
          """
        }
        store_notification_processed_already(
          "${workflow.start}-already_processed_fastq_samples.txt",
          ch_fastq_sample_qc_pass_matches_search.matched.collect()
        )

        ch_fasta_matching_metadata.ifEmpty {
            log.error """\
              ERROR: No file was found for samples in metadata.
                This may be caused by a failure in loading samples from the metadata file to the database.
                Aborting!
            """
            System.exit(1)
        }

    emit:
        ch_fasta_matching_metadata
}

// Process which acts as a workaround to append boolean value, whether the sample exists in another channel
process append_metadata_match_to_sample_file_pair{
  input:
    tuple val(sample_name), file(fastq1), file(fastq2)
    val samples_with_meta

  output:
    tuple val(sample_name), file(fastq1), file(fastq2), val(has_meta)

  script:
    has_meta = samples_with_meta.contains(sample_name)

  """
  """
}

process store_fastas_not_matched{
    publishDir COVID_PIPELINE_MISSING_METADATA_PATH, mode: 'copy', overwrite: true

    input:
      file(fastq)

    output:
      file(fastq)

    script:

    """
    """
}
