include { organise_metadata_sample_files } from './common/organise_metadata_sample_files.nf'
include { run_analysis } from './run_analysis.nf'
include { generate_synthetic_output } from './generate_synthetic_output.nf'


/*
 * Main workflow for the pathogen: dummy_pathogen.
 */
workflow psga {

    main:
        organise_metadata_sample_files()
        ch_metadata = organise_metadata_sample_files.out.ch_metadata
        ch_sample_files = organise_metadata_sample_files.out.ch_sample_files

        run_analysis(ch_sample_files)

        generate_synthetic_output(
            ch_metadata,
            run_analysis.out.ch_input_files.collect()
        )

        ch_analysis_run_results_submitted = generate_synthetic_output.out.ch_output_csv_file

    emit:
        ch_analysis_run_results_submitted
}
