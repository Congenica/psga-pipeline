include { makeFastqSearchPath } from './utils.nf'

include { append_metadata_match_to_sample_file_pair }  from './utils.nf'
include { store_mismatching_files }  from './utils.nf'
include { concat_elements_to_single_string as concat_metadata_samples } from './utils.nf'
include { concat_elements_to_single_string as concat_fastq_samples } from './utils.nf'

include { append_match_to_values_list as append_match_to_samples }  from './utils.nf'
include { append_match_to_values_list as append_qc_pass_match_to_sample_files }  from './utils.nf'

include { notifications } from './filtering.nf'


workflow filter_fastq_matching_with_metadata{
    take:
        ch_all_samples_with_metadata_loaded
        ch_samples_with_metadata_loaded
        ch_qc_passed_samples
        ch_updated_samples
    main:
        // Original usage of grouping illumina fastq file pairs and extracting sample name
        ch_file_paths = makeFastqSearchPath(
            params.illuminaPrefixes,
            params.illuminaSuffixes,
            params.fastq_exts
        )
        Channel.fromFilePairs( ch_file_paths, flat: true)
            .filter{ !( it[0] =~ /Undetermined/ ) }
            .set{ ch_sample_files }
        // ch_sample_files: [[sample_name, fastq1, fastq2], ...]

        ch_sample_files
            .map { it ->  it[0] }
            .flatten()
            .set{ ch_sample_names }

        // Matching our extracted sample names with samples loaded from the metadata
        ch_sample_files_matches = append_metadata_match_to_sample_file_pair(
            ch_sample_files,
            ch_all_samples_with_metadata_loaded.collect()
        )

        ch_sample_files_matches
            .branch { it ->
                matched: it[3]
                    return it
                mismatched: true
                    return it
            }
            .set{ ch_sample_files_grouped_by_metadata_search }

        ch_sample_files_grouped_by_metadata_search.matched
             .map { it -> [ it[1], it[2] ] }
             .flatten()
             .set{ ch_files_matching_metadata }

        ch_sample_files_grouped_by_metadata_search.mismatched
             .map { it -> [ it[1], it[2] ] }
             .flatten()
             .set{ ch_files_mismatching_metadata }
        ch_sample_files_grouped_by_metadata_search.mismatched
             .map { it -> it[0] }
             .flatten()
             .set{ ch_mismatching_metadata_sample_names }

        store_mismatching_files(ch_files_mismatching_metadata)

        ch_metadata_sample_matches = append_match_to_samples(
            ch_samples_with_metadata_loaded,
            ch_sample_names.collect()
        )

        ch_metadata_sample_matches
            .branch { it ->
                matched: it[1]
                    return it[0]
                mismatched: true
                    return it[0]
            }
            .set{ ch_metadata_sample_matches_search }

        ch_sample_qc_pass_matches = append_qc_pass_match_to_sample_files(
            ch_sample_names,
            ch_qc_passed_samples.collect()
        )

        ch_sample_qc_pass_matches
            .branch { it ->
                matched: it[1]
                    return it[0]
                mismatched: true
                    return it[0]
            }
            .set{ ch_sample_qc_pass_matches_search }

        notifications(
          ch_metadata_sample_matches_search.mismatched,
          ch_updated_samples,
          ch_mismatching_metadata_sample_names,
          ch_sample_qc_pass_matches_search.matched,
          ch_files_matching_metadata
        )

        ch_selected_sample_files = notifications.out.ch_files_matching_metadata
        store_notification_missing_files = notifications.out.store_notification_missing_files

    emit:
        ch_selected_sample_files
        store_notification_missing_files
}
