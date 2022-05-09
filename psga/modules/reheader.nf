/*
 * Reheader the genome fasta files generated by ncov2019-artic-nf
 */
process reheader_fasta {
  tag "${task.index} - ${ncov_fasta}"
  input:
    path ncov_fasta

  output:
    path "*.fasta"

  script:
    files_dir = "./"
    output_dir = "./"

  """
  python ${PSGA_ROOT_PATH}/scripts/reheader_fasta.py ${files_dir} ${output_dir}
  """
}

/*
 * Process to store fastas, which were marked in ncov pipeline as QC_PASS=TRUE
 */
process store_qc_passed_fasta {
  tag "${task.index} - ${reheadered_fasta_file}"
  publishDir "${PSGA_OUTPUT_PATH}/reheadered-fasta", mode: 'copy', overwrite: true

  input:
    path reheadered_fasta_file
    val sample_name

  output:
    path matching_file, emit: ch_qc_passed_fasta

  script:
    matching_file = "${sample_name}.fasta"

  """
  """
}

/*
 * Process to store fastas, which were marked in ncov pipeline as QC_PASS=FALSE
 */
process store_qc_failed_fasta {
  tag "${task.index} - ${reheadered_fasta_file}"
  publishDir "${PSGA_OUTPUT_PATH}/reheadered-fasta-qc-failed", mode: 'copy', overwrite: true

  input:
    path reheadered_fasta_file
    val sample_name

  output:
    path matching_file, emit: ch_qc_failed_fasta

  script:
    matching_file = "${sample_name}.fasta"

  """
  """
}

/*
 * Reheader a fasta file and store it based on ncov QC
 */
workflow reheader {
    take:
        ch_ncov_qc
        ch_ncov_fasta
    main:

        ch_reheadered_fasta = reheader_fasta(ch_ncov_fasta)

        // define whether sample is QC_PASSED or QC_FAILED
        ch_ncov_qc
            .splitCsv(header:true)
            .branch {
                qc_passed: it.qc_pass =~ /TRUE/
                    return it.sample_name
                qc_failed: true
                    return it.sample_name
            }
            .set{ ch_sample_row_by_qc }

        ch_qc_passed_fasta = store_qc_passed_fasta(
            ch_reheadered_fasta,
            ch_sample_row_by_qc.qc_passed.flatten()
        )
        store_qc_failed_fasta(
            ch_reheadered_fasta,
            ch_sample_row_by_qc.qc_failed.flatten()
        )

    emit:
        ch_qc_passed_fasta
}
