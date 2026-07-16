import gzip
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts/build_pretrain_10b.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_pretrain_10b",
    SCRIPT_PATH,
)
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


class ProvenancePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.journal = self.root / "code-provenance.jsonl"
        self.archive = self.root / "code-provenance.jsonl.gz"

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def record(source, hash_value):
        return {
            "hash_xxh3_64": hash_value,
            "source": source,
            "hexsha": "abc123",
            "lang": "Java",
            "ext": "java",
            "max_stars_repo_name": "example/repository",
            "max_stars_repo_licenses": ["apache-2.0"],
        }

    @staticmethod
    def encoded_records(records):
        return b"".join(
            (
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8")
            for record in records
        )

    def write_archive(self, records, path=None):
        path = path or self.archive

        with gzip.open(path, "wb") as handle:
            handle.write(
                self.encoded_records(records)
            )

    def test_valid_gzip_migrates_to_journal(self):
        records = [
            self.record("code-java", "0000000000000001"),
            self.record("code-java", "0000000000000002"),
        ]
        expected = self.encoded_records(records)
        self.write_archive(records)

        result = build.prepare_provenance_journal(
            self.journal,
            self.archive,
        )

        self.assertEqual(result, self.journal)
        self.assertEqual(self.journal.read_bytes(), expected)
        self.assertEqual(
            build.validate_provenance_journal(self.journal),
            2,
        )

    def test_journal_resumes_without_gzip_append(self):
        records = [
            self.record("code-java", "0000000000000001"),
            self.record("code-java", "0000000000000002"),
        ]
        self.journal.write_bytes(
            self.encoded_records(records[:1])
        )
        self.write_archive(records[:1])
        archive_before = self.archive.read_bytes()

        build.prepare_provenance_journal(
            self.journal,
            self.archive,
        )
        journal = build.ProvenanceJournal(self.journal)

        try:
            journal.write_record(records[1])
            journal.flush()
            os.fsync(journal.fileno())
        finally:
            journal.close()

        self.assertEqual(
            self.journal.read_bytes(),
            self.encoded_records(records),
        )
        self.assertEqual(
            self.archive.read_bytes(),
            archive_before,
        )

    def test_trailing_partial_record_is_quarantined_and_truncated(self):
        record = self.record(
            "code-java",
            "0000000000000001",
        )
        complete = self.encoded_records([record])
        partial = b'{"source": "code-java"'
        self.journal.write_bytes(
            complete + partial
        )

        build.prepare_provenance_journal(
            self.journal,
            self.archive,
        )

        self.assertEqual(
            self.journal.read_bytes(),
            complete,
        )
        quarantine_files = list(
            self.root.glob(
                "code-provenance.jsonl.quarantine-*"
            )
        )
        self.assertEqual(len(quarantine_files), 1)
        self.assertEqual(
            quarantine_files[0].read_bytes(),
            partial,
        )

        journal = build.ProvenanceJournal(self.journal)

        try:
            journal.write_record(
                self.record(
                    "code-java",
                    "0000000000000002",
                )
            )
        finally:
            journal.close()

        self.assertEqual(
            build.validate_provenance_journal(self.journal),
            2,
        )

    def test_invalid_interior_record_fails(self):
        records = [
            self.record("code-java", "0000000000000001"),
            self.record("code-java", "0000000000000002"),
        ]
        original = (
            self.encoded_records(records[:1])
            + b"not-json\n"
            + self.encoded_records(records[1:])
        )
        self.journal.write_bytes(original)

        with self.assertRaises(ValueError):
            build.prepare_provenance_journal(
                self.journal,
                self.archive,
            )

        self.assertEqual(
            self.journal.read_bytes(),
            original,
        )

    def test_atomic_archive_publish(self):
        records = [
            self.record("code-java", "0000000000000001"),
            self.record("code-java", "0000000000000002"),
        ]
        expected = self.encoded_records(records)
        self.journal.write_bytes(expected)

        count = build.publish_provenance_archive(
            self.journal,
            self.archive,
        )

        self.assertEqual(count, 2)
        self.assertEqual(
            build.validate_provenance_archive(
                self.archive,
                expected_count=2,
            ),
            2,
        )

        with gzip.open(self.archive, "rb") as handle:
            self.assertEqual(handle.read(), expected)

    def test_corrupt_existing_gzip_refuses_migration(self):
        records = [
            self.record("code-java", "0000000000000001"),
        ]
        self.write_archive(records)
        corrupt = self.archive.read_bytes()[:-8]
        self.archive.write_bytes(corrupt)

        with self.assertRaises((EOFError, OSError)):
            build.prepare_provenance_journal(
                self.journal,
                self.archive,
            )

        self.assertFalse(self.journal.exists())
        self.assertEqual(
            self.archive.read_bytes(),
            corrupt,
        )

    def test_interrupted_archive_publish_preserves_old_archive(self):
        old_records = [
            self.record("code-java", "0000000000000001"),
        ]
        new_records = old_records + [
            self.record("code-java", "0000000000000002"),
        ]
        self.write_archive(old_records)
        archive_before = self.archive.read_bytes()
        self.journal.write_bytes(
            self.encoded_records(new_records)
        )

        with mock.patch.object(
            build,
            "validate_provenance_archive",
            side_effect=RuntimeError("simulated interruption"),
        ):
            with self.assertRaises(RuntimeError):
                build.publish_provenance_archive(
                    self.journal,
                    self.archive,
                )

        self.assertEqual(
            self.archive.read_bytes(),
            archive_before,
        )
        self.assertEqual(
            list(self.root.glob("*.tmp-*")),
            [],
        )

    def test_post_replace_sync_failure_preserves_old_archive(self):
        old_records = [
            self.record("code-java", "0000000000000001"),
        ]
        new_records = old_records + [
            self.record("code-java", "0000000000000002"),
        ]
        self.write_archive(old_records)
        archive_before = self.archive.read_bytes()
        self.journal.write_bytes(
            self.encoded_records(new_records)
        )

        with mock.patch.object(
            build,
            "fsync_directory",
            side_effect=[
                None,
                OSError("simulated directory fsync failure"),
                None,
            ],
        ):
            with self.assertRaises(OSError):
                build.publish_provenance_archive(
                    self.journal,
                    self.archive,
                )

        self.assertEqual(
            self.archive.read_bytes(),
            archive_before,
        )
        self.assertEqual(
            list(self.root.glob("*.previous-*")),
            [],
        )

    def test_migration_does_not_introduce_duplicate_records(self):
        records = [
            self.record("code-java", "0000000000000001"),
            self.record("code-java", "0000000000000002"),
        ]
        expected = self.encoded_records(records)
        self.write_archive(records)

        build.prepare_provenance_journal(
            self.journal,
            self.archive,
        )
        build.prepare_provenance_journal(
            self.journal,
            self.archive,
        )

        self.assertEqual(
            self.journal.read_bytes(),
            expected,
        )
        self.assertEqual(
            build.validate_provenance_journal(self.journal),
            2,
        )


if __name__ == "__main__":
    unittest.main()
