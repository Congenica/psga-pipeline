-- Verify ps-bahrain-covid:10-add_metadata_loaded_flag on pg

BEGIN;

    SET LOCAL search_path = sars_cov_2;

    SELECT has_table_privilege('sample', 'SELECT, INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES');

ROLLBACK;
