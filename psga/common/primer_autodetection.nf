/*
 * Run: primer-autodetection (use first read only for illumina)
 */
process primer_autodetection {
  publishDir "${params.output_path}/primer_autodetection", mode: 'copy', overwrite: true, pattern: '{*_primer_data.csv,*_primer_detection.csv,trimmomatic.out,*.bowtie2*}'

  tag "${task.index} - ${fastq}"

  input:
    path fastq

  output:
    tuple path("*_primer.txt"), path(fastq), emit: ch_files
    path "*_primer_data.csv", emit: ch_primer_data
    path "*_primer_detection.csv", emit: ch_primer_coverage
    path "trimmomatic.out"
    path "*.bowtie2*"

  shell:
  '''
  # select the input fastq file based on illumina or ont.
  # For illumina, 1 read is sufficient for primer autodetection
  if [[ "!{params.sequencing_technology}" == "illumina" ]]; then
    file_1="$(ls *_1.fastq.gz)"
  elif [[ "!{params.sequencing_technology}" == "ont" ]]; then
    file_1="!{fastq}"
  else
    echo "sequencing technology must be either 'illumina' or 'ont' for primer autodetection"
    exit 1
  fi

  # extract the sample id, whereas this is ID.fastq.gz or ID_1.fastq.gz
  sample_id=$( echo ${file_1} | cut -d '.' -f1 | cut -d '_' -f1)
  trimmed_sample="trimmed_sample.fastq.gz"
  crop="!{params.primer_minimum_length}"

  # trim sample reads so that they are long as the shortest primer
  trimmomatic SE -phred33 ${file_1} ${trimmed_sample} CROP:${crop} 2> trimmomatic.out

  # iterate over the available primer scheme fasta.
  # These fasta are generated by us and contain the sequences for each primer scheme name/version
  readarray -t primer_fasta_array < "/primer_schemes/SARS-CoV-2_primer_fasta_index.txt"
  for primer_fasta in "${primer_fasta_array[@]}"; do
      # extract the primer (e.g. ARTIC_V3)
      primer="$(echo "${primer_fasta}" | cut -d/ -f3,5 | sed 's#/#_#')"

      # build the index for the primer fasta
      bowtie2-build ${primer_fasta} ${primer}.ref_idx &> ${primer}.bowtie2-build

      # end-to-end align the trimmed sample to the primer scheme fasta
      bowtie2 --no-unal -x ${primer}.ref_idx -U ${trimmed_sample} 2> ${primer}.bowtie2 | samtools sort -o ${primer}.bam

      # extract the coverage and other metrics from the bam. `samtools coverage` complains if the bam is not sorted
      samtools coverage ${primer}.bam > ${primer}.coverage.tsv
  done

  python ${PSGA_ROOT_PATH}/scripts/common/primer_autodetection.py --input-path . --sample-id "${sample_id}" --primer-input !{params.kit}
  '''
}
