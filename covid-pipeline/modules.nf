/*
 * Load the I-SEHA metadata into the database
 */
process load_iseha_metadata {
  input:
    path ch_iseha_metadata_tsv_file

  output:
    path iseha_metadata_load_done, emit: ch_iseha_metadata_load_done
    path samples_with_metadata, emit: ch_samples_with_metadata_file

  script:
    iseha_metadata_load_done = "load_iseha_metadata.done"
    samples_with_metadata = "all_samples_with_metadata.txt"

  """
  python /app/scripts/load_iseha_metadata.py \
    --file "${ch_iseha_metadata_tsv_file}" \
    --output_file_for_samples_with_metadata "${samples_with_metadata}"
  touch ${iseha_metadata_load_done}
  """
}

/*
 * Run: ncov2019-artic-nf nextflow pipeline
 * see: https://github.com/connor-lab/ncov2019-artic-nf
 */
process ncov2019_artic_nf_pipeline {
  publishDir COVID_PIPELINE_WORKDIR, mode: 'copy', overwrite: true

  input:
    file fastq_file
    val ncov_docker_image
    val ncov_prefix

  output:
    path "${ncov_out_directory}/*", emit: ch_all_ncov_results
    path "${ncov_out_directory}/ncovIllumina_sequenceAnalysis_makeConsensus/*.fa", emit: ch_fasta_ncov_results
    path "${ncov_out_directory}/${ncov_prefix}.qc.csv", emit: ch_qc_csv_ncov_result
    path "${ncov_out_directory}/qc_plots/*.depth.png", emit: ch_sample_depth_ncov_results

  script:
    ncov_out_directory = "ncov_output"

  """
  nextflow run ${COVID_PIPELINE_ROOTDIR}/ncov2019-artic-nf -profile docker --illumina --prefix ${ncov_prefix} --directory `eval pwd` -with-docker ${ncov_docker_image} --outdir ${ncov_out_directory}
  """
}

/*
 * Load ncov2019-artic-nf assembly qc data to the database
 */
process load_ncov_assembly_qc_to_db {
  input:
    file ch_qc_ncov_result_csv_file
    file ch_qc_plot_files

  output:
    path ch_ncov_qc_load_done, emit: ch_ncov_qc_load_done

  script:
    directory_with_qc_depth_files = "./"
    ch_ncov_qc_load_done = "load_ncov_assembly_qc_to_db.done"

  """
  python /app/scripts/submit_sample_qc.py \
    --ncov_qc_csv_file "${ch_qc_ncov_result_csv_file}" \
    --ncov_qc_depth_directory "${directory_with_qc_depth_files}" \
    --pipeline_version "${workflow.manifest.version}"
  touch ${ch_ncov_qc_load_done}
  """
}

/*
 * Reheader the genome fasta files generated by ncov2019-artic-nf
 */
process reheader_genome_fasta {
  input:
    path ncov_fasta

  output:
    path "*.fasta"

  script:
    files_dir = "./"
    output_dir = "./"

  """
  python /app/scripts/reheader_fasta.py ${files_dir} ${output_dir}
  """
}

/*
 * Process to store fastas, which were marked in ncov pipeline as QC_PASS=TRUE
 */
process store_reheadered_fasta_passed {
  publishDir COVID_PIPELINE_FASTA_PATH, mode: 'copy', overwrite: true

  input:
    path all_reheadered_fasta_files
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
process store_reheadered_fasta_failed {
  publishDir COVID_PIPELINE_FASTA_PATH_QC_FAILED, mode: 'copy', overwrite: true

  input:
    path all_reheadered_fasta_files
    val sample_name

  output:
    path matching_file, emit: ch_qc_failed_fasta

  script:
    matching_file = "${sample_name}.fasta"

  """
  """
}

/*
 * Concatenate reheadered fasta files for Nextstrain pipeline
 */
process concatenate_fasta {
  input:
    val root_genome_fasta
    path reheadered_fasta
    path archived_fasta

  output:
    path output_file

  script:
    files_dir = "./"
    output_file = "nextstrain.fasta"

  """
  # create links in ${files_dir} to the archived files, so that these can be concatenated
  python /app/scripts/link_archived_fasta.py --destination ${files_dir}

  echo "FASTA files to concatenate:"
  ls -l

  python /app/scripts/concatenate_fasta.py --output ${output_file} --root-genome ${root_genome_fasta} ${files_dir}
  """
}

/*
 * Run: pangolin snakemake pipeline
 * see: https://github.com/cov-lineages/pangolin
 */
process pangolin_pipeline {
  publishDir COVID_PIPELINE_WORKDIR, mode: 'copy', overwrite: true

  input:
    path reheadered_fasta

  output:
    tuple val(sample_name), path("${pangolin_out_directory}/${output_filename}"), emit: ch_pangolin_lineage_csv

  script:
    sample_name = reheadered_fasta.getSimpleName()
    pangolin_out_directory = "pangolin_output"
    output_filename = "${sample_name}_lineage_report.csv"
  """
  pangolin ${reheadered_fasta} --outdir ${pangolin_out_directory} --outfile ${output_filename}
  """
}

/*
 * Load pangolin data to the database
 */
process load_pangolin_data_to_db {
  input:
    tuple val(sample_name), file(ch_pangolin_result_csv_file)

  output:
    path ch_pangolin_load_data_done

  script:
    ch_pangolin_load_data_done = "${sample_name}.load_pangolin_data_to_db.done"

  """
  python /app/scripts/load_pangolin_data_to_db.py \
    --pangolin-lineage-report-file "${ch_pangolin_result_csv_file}" \
    --sample-name "${sample_name}"
  touch ${ch_pangolin_load_data_done}
  """
}

/*
 * Prepare metadata tsv file, which is used as an input for nextstrain pipeline
 */
process prepare_tsv_for_nextstrain {
  input:
    path ncov_qc_to_db_submit_completion_flag
    path pangolin_to_db_submit_completion_flag

  output:
    path nextstrain_analysis_tsv

  script:
    nextstrain_analysis_tsv = "nextstrain_metadata.tsv"

  """
  python /app/scripts/generate_nextstrain_input_tsv.py --output ${nextstrain_analysis_tsv}
  """
}

process generate_report_strain_level_and_global_context {
  publishDir COVID_PIPELINE_REPORTS_PATH, mode: 'copy', overwrite: true

  input:
    val pangolearn_lineage_notes_url
    val pangolearn_metadata_url
    val pangolearn_dir
    path pangolin_to_db_submit_completion_flag

  output:
    path output_filename

  script:
    report_name = "strain_level_and_global_context"
    output_filename = "${report_name}_report.csv"

  """
  python /app/scripts/generate_report.py \
    --report "${report_name}" \
    --lineage-notes-url "${pangolearn_lineage_notes_url}" \
    --metadata-url "${pangolearn_metadata_url}" \
    --pangolearn-dir "${pangolearn_dir}" \
    --output "${output_filename}"
  """
}

process generate_report_strain_first_seen {
  publishDir COVID_PIPELINE_REPORTS_PATH, mode: 'copy', overwrite: true

  input:
    path pangolin_to_db_submit_completion_flag

  output:
    path output_filename

  script:
    report_name = "strain_first_seen"
    output_filename = "${report_name}_report.csv"

  """
  python /app/scripts/generate_report.py \
    --report "${report_name}" \
    --output "${output_filename}"
  """
}

/*
 * Prepare input tsv for microreact
 */
process prepare_microreact_tsv {
  input:
    path ncov_qc_to_db_submit_completion_flag
    path pangolin_to_db_submit_completion_flag

  output:
    path params.microreact_tsv

  """
  python /app/scripts/generate_microreact_input.py --output ${params.microreact_tsv}
  """
}

/*
 * Run: nextstrain-ncov snakemake pipeline
 * see: https://github.com/nextstrain/ncov
 */
process nextstrain_pipeline {
  publishDir COVID_PIPELINE_WORKDIR, mode: 'copy', overwrite: true

  input:
    path metadata
    path fasta

  output:
    path "${nextstrain_out_directory}/*", emit: ch_all_nextstrain_results
    path "${nextstrain_out_directory}/bahrain/ncov_with_accessions.json", emit: ch_nextstrain_ncov_with_accessions_json
    path "${nextstrain_out_directory}/bahrain/aa_muts.json", emit: ch_nextstrain_aa_muts_json
    path "${nextstrain_out_directory}/bahrain/nt_muts.json", emit: ch_nextstrain_nt_muts_json
    path "${nextstrain_out_directory}/bahrain/tree.nwk", emit: ch_nextstrain_tree_nwk

  script:
    nextstrain_out_directory = "nextstrain_output"
  """
  # The snakemake custom profile is configured so that the input data are in: /nextstrain/data/
  # These copies must be executed as root as they alter the container, hence --user 0:0 in nextflow.config.
  # note: ln -s does not work here
  cp ${metadata} /nextstrain/data/nextstrain_metadata.tsv
  cp ${fasta} /nextstrain/data/nextstrain.fasta

  # create the output dir. This must be done on /
  mkdir -p ${nextstrain_out_directory}
  cd /nextstrain
  # run the nextstrain Snakemake pipeline
  snakemake --profile /custom_profile

  # copy the output files to the caller and set user permissions. This must be done on /
  # note: ln -s does not work here
  cd -
  cp -R /nextstrain/results/* ${nextstrain_out_directory}
  chown -R \${UID}:\${GID} ${nextstrain_out_directory}
  """
}

/*
 * Load Nextstrain data to the database.
 * Data are: amino acid and nucleotide mutations.
 */
process load_nextstrain_data_to_db {
  input:
    file(ch_nextstrain_aa_muts_json_file)
    file(ch_nextstrain_nt_muts_json_file)
    file(ch_nextstrain_tree_nwk_file)

  output:
    path ch_load_nextstrain_data_done, emit: ch_nextstrain_data_load_done

  script:
    ch_load_nextstrain_data_done = "load_nextstrain_data_to_db.done"

  """
  # load amino acid mutations
  python /app/scripts/load_nextstrain_aa_muts_to_db.py \
    --aa-muts-json "${ch_nextstrain_aa_muts_json_file}" \
    --tree-nwk "${ch_nextstrain_tree_nwk_file}"

  # load nucleotide mutations
  python /app/scripts/load_nextstrain_nt_muts_to_db.py \
    --nt-muts-json "${ch_nextstrain_nt_muts_json_file}" \
    --tree-nwk "${ch_nextstrain_tree_nwk_file}"

  touch ${ch_load_nextstrain_data_done}
  """
}
