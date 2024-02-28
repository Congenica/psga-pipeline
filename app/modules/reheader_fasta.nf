/*
 * Reheader a fasta file
 */
process REHEADER_FASTA {
  publishDir "${params.output_path}", mode: 'copy', overwrite: true, pattern: 'reheadered_fasta/*.fasta'

  input:
      tuple val(meta), path(fasta)

  output:
      tuple val(meta), path("reheadered_fasta/*.fasta")

  shell:
    '''
    # standardise input fasta before re-headering
    fasta=!{fasta}
    mkdir input_dir reheadered_fasta
    cp ${fasta} input_dir/${fasta%.*}.consensus.fa

    python ${PSGA_ROOT_PATH}/scripts/ncov/reheader_fasta.py --input-dir input_dir --output-dir reheadered_fasta
    '''
}
