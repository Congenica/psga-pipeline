/*
 * Generate files, required for upload to GenBank
 */
process create_genbank_submission_files {
  input:
    file reheadered_fasta
    file archived_fasta
    file genbank_submission_template
    val genbank_submission_comment
    val genbank_submitter_name
    val genbank_submitter_account_namespace
    val genbank_submission_id_suffix

  output:
    path submission_xml, emit: ch_genbank_xml
    path submission_zip, emit: ch_genbank_zip
    path submission_samples_txt, emit: ch_samples_txt
    val unique_genbank_submission_identifier, emit: ch_genbank_submission_id

  script:
    sequence_fasta_directory = "./"
    sequence_data_fsa = "sequence.fsa"
    source_metadata_table_src = "source.src"
    submission_xml = "submission.xml"
    submission_zip = "submission.zip"
    submission_samples_txt = "samples_in_submission.txt"
    unique_genbank_submission_identifier = "${workflow.sessionId}.${genbank_submission_id_suffix}"

  """
  # create links in ${sequence_fasta_directory} to the archived files, so that these can be concatenated
  python /app/scripts/link_archived_fasta.py --destination ${sequence_fasta_directory}

  echo "All FASTA files to submit to GenBank:"
  ls -l

  python /app/scripts/generate_genbank_files.py \
    --input_sequence_fasta_directory ${sequence_fasta_directory} \
    --input_submission_template ${genbank_submission_template} \
    --output_sequence_data_fsa ${sequence_data_fsa} \
    --output_source_metadata_table_src ${source_metadata_table_src} \
    --output_submission_xml ${submission_xml} \
    --output_submission_zip ${submission_zip} \
    --output_samples_submitted_file ${submission_samples_txt} \
    --submit_name "${genbank_submission_comment}" \
    --submitter "${genbank_submitter_name}" \
    --spuid_namespace "${genbank_submitter_account_namespace}" \
    --spuid_unique_value "${unique_genbank_submission_identifier}"
  """
}

/*
 *  Submit GenBank files to FTP
 */
process submit_genbank_files {
  input:
    path submission_xml
    path submission_zip
    path sample_names_txt
    val submit_id
    val no_samples_flag
    val remote_url
    val remote_username
    val remote_password

  when:
    no_samples_flag != ['NO_SAMPLES']

  output:
    path submission_xml, emit: ch_genbank_xml
    path submission_zip, emit: ch_genbank_zip
    path sample_names_txt, emit: ch_genbank_sample_names_txt
    val submit_id, emit: ch_genbank_submission_id

  script:
    ch_submit_genbank_files_done = "submit_genbank_files.done"

  """
  echo ${no_samples_flag}

  python /app/scripts/submit_genbank_files.py \
    --input_xml ${submission_xml} \
    --input_zip ${submission_zip} \
    --submit_id ${submit_id} \
    --url ${remote_url} \
    --username ${remote_username} \
    --password ${remote_password}

  touch ${ch_submit_genbank_files_done}
  """
}

/*
 *  Mark samples submitted with GenBank identifier, so that samples are not submitted in next runs
 */
process mark_samples_as_submitted_to_genbank{
  input:
    file sample_names_txt
    val no_samples_flag
    val submit_id

  when:
    no_samples_flag != ['NO_SAMPLES']

  output:
    file mark_samples_as_submitted_to_genbank_done

  script:
    mark_samples_as_submitted_to_genbank_done = "mark_samples_as_submitted_to_genbank.done"

  """
  python /app/scripts/mark_submitted_genbank_samples.py \
    --sample_names_txt "${sample_names_txt}" \
    --submit_id "${submit_id}"

  touch ${mark_samples_as_submitted_to_genbank_done}
  """
}

/*
 *  Publish GenBank submission in archive directory.
 */
process store_genbank_submission{
  publishDir COVID_PIPELINE_GENBANK_PATH, mode: 'copy', overwrite: true

  input:
    path submission_xml
    path submission_zip
    path sample_names_file
    val submit_id

  output:
    path "${out_directory}/*", emit: ch_genbank_submit_files

  script:
    out_directory = submit_id

  """
  mkdir "${out_directory}"
  mv "${submission_xml}" "${out_directory}/${submission_xml}"
  mv "${submission_zip}" "${out_directory}/${submission_zip}"
  mv "${sample_names_file}" "${out_directory}/${sample_names_file}"
  """
}
