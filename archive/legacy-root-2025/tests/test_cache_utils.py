import os
import hashlib
from trail_route_ai import cache_utils


import unittest
import shutil
# hashlib and os are already imported at the top of the file
# from trail_route_ai import cache_utils is also effectively imported


class TestCacheUtilsRocksDB(unittest.TestCase):

    def setUp(self):
        # Ensure the default cache directory exists for testing
        self.test_cache_dir_base = cache_utils.DEFAULT_CACHE_DIR # Use the default base
        # Define a specific subdirectory for these tests to avoid conflicts
        self.rocksdb_test_specific_dir = os.path.join(self.test_cache_dir_base, "test_rocksdb_dbs_temp")

        # Clean up any previous test runs
        if os.path.exists(self.rocksdb_test_specific_dir):
            shutil.rmtree(self.rocksdb_test_specific_dir)
        os.makedirs(self.rocksdb_test_specific_dir, exist_ok=True)

        # Override get_cache_dir for the duration of the tests
        # Store the original function
        self._original_get_cache_dir = cache_utils.get_cache_dir
        # Create a lambda that returns the specific test directory
        cache_utils.get_cache_dir = lambda: self.rocksdb_test_specific_dir

        self.db_name = "test_db_ops" # Renamed to avoid potential clashes
        self.db_key = "test_key_ops"
        # Construct the expected path for verification and cleanup
        h = hashlib.sha1(self.db_key.encode()).hexdigest()[:16]
        self.expected_db_path = os.path.join(self.rocksdb_test_specific_dir, f"{self.db_name}_{h}_db")


    def tearDown(self):
        # Restore the original get_cache_dir function
        cache_utils.get_cache_dir = self._original_get_cache_dir

        # Clean up the specific test directory for RocksDBs
        if os.path.exists(self.rocksdb_test_specific_dir):
            shutil.rmtree(self.rocksdb_test_specific_dir)
        # Just in case, if the db_path was somehow created outside the specific dir by mistake
        if os.path.exists(self.expected_db_path): # Check before attempting to remove
             shutil.rmtree(self.expected_db_path)


    def test_open_rocksdb_read_only_non_existent(self):
        # Attempt to open a non-existent DB in read-only mode
        db = cache_utils.open_rocksdb(self.db_name, self.db_key, read_only=True)
        self.assertIsNone(db, "Opening a non-existent DB in read-only mode should return None.")
        self.assertFalse(os.path.exists(self.expected_db_path), "DB should not be created in read-only non-existent case.")

    def test_open_rocksdb_read_write_create_and_read(self):
        # Open in read-write mode, which should create it
        db_rw = cache_utils.open_rocksdb(self.db_name, self.db_key, read_only=False)
        self.assertIsNotNone(db_rw, "Failed to open/create DB in read-write mode.")
        self.assertTrue(os.path.exists(self.expected_db_path), "DB path should exist after read-write open.")

        try:
            # Perform a simple write and read
            db_rw[b"test_key_rw"] = b"test_value_rw"
            retrieved_value = db_rw[b"test_key_rw"]
            self.assertEqual(retrieved_value, b"test_value_rw", "Failed to read/write to the DB.")
        finally:
            cache_utils.close_rocksdb(db_rw)

    def test_open_rocksdb_read_only_existing(self):
        # First, create a DB by opening in read-write mode
        db_setup = cache_utils.open_rocksdb(self.db_name, self.db_key, read_only=False)
        self.assertIsNotNone(db_setup, "Setup: Failed to create DB for read-only test.")
        try:
            db_setup[b"sample_key"] = b"sample_value"
        finally:
            cache_utils.close_rocksdb(db_setup)

        # Now, open the existing DB in read-only mode
        db_ro = cache_utils.open_rocksdb(self.db_name, self.db_key, read_only=True)
        self.assertIsNotNone(db_ro, "Failed to open existing DB in read-only mode.")

        try:
            # Try to read from it
            value = db_ro[b"sample_key"]
            self.assertEqual(value, b"sample_value", "Failed to read from existing DB in read-only mode.")

            # Attempting a write should fail or not persist if truly read-only
            # (Further notes on read-only behavior as in original prompt)
        finally:
            cache_utils.close_rocksdb(db_ro)

    def test_close_rocksdb_raises(self):
        class BadDB:
            def close(self):
                raise OSError("boom")

        with self.assertRaises(OSError):
            cache_utils.close_rocksdb(BadDB())
