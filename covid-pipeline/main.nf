#!/usr/bin/env nextflow

// Enable DSL 2 syntax
nextflow.enable.dsl = 2

log.info """\
    ======================
    COVID pipeline v 1.0.0
    ======================
    ncov2019-artic-nf config:
    * fastq sample dir : $params.ncov_fastq_sample_dir
    * project dir      : $params.ncov_pipeline_dir
    * docker image     : $params.ncov_docker_image
    * prefix           : $params.ncov_prefix

    env vars:
    * COVID_PIPELINE_WORKDIR : ${COVID_PIPELINE_WORKDIR}
    * GENOME_FASTA_PATH      : ${GENOME_FASTA_PATH}
    ======================
"""


// Import modules
include { ncov2019_artic_nf_pipeline } from './modules.nf'
include { reheader_genome_fasta } from './modules.nf'
include { pangolin_pipeline } from './modules.nf'


workflow {

    ncov2019_artic_nf_pipeline(
        params.ncov_pipeline_dir,
        params.ncov_docker_image,
        params.ncov_prefix,
        params.ncov_fastq_sample_dir
    )

    // flatten the resulting fasta, so that pipeline branches off per-fasta to its own separate processes
    ncov2019_artic_nf_pipeline.out.ch_fasta_ncov_results \
        .flatten() \
        .set { ch_fasta_to_reheader }
    ch_reheadered_fasta = reheader_genome_fasta(ch_fasta_to_reheader)

    pangolin_pipeline(
        ch_reheadered_fasta
    )
}
