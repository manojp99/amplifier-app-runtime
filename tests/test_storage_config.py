"""Tests for configurable session storage."""
import os
import tempfile
from pathlib import Path

import pytest

from amplifier_app_runtime.session import SessionManager
from amplifier_app_runtime.session_store import SessionStore


class TestStorageConfiguration:
    """Test session storage configuration options."""

    def test_default_behavior(self):
        """Ensure default behavior unchanged."""
        # Clear any env vars
        os.environ.pop("AMPLIFIER_STORAGE_DIR", None)
        os.environ.pop("AMPLIFIER_NO_PERSIST", None)
        
        manager = SessionManager()
        assert manager._store is not None
        assert "/.amplifier/projects/" in str(manager._store.storage_dir)

    def test_custom_storage_dir(self):
        """Test custom storage directory via environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["AMPLIFIER_STORAGE_DIR"] = tmpdir
            os.environ.pop("AMPLIFIER_NO_PERSIST", None)
            
            manager = SessionManager()
            assert manager._store is not None
            assert str(manager._store.storage_dir) == tmpdir
            
            # Cleanup
            os.environ.pop("AMPLIFIER_STORAGE_DIR", None)

    def test_no_persist(self):
        """Test persistence disabled via environment variable."""
        os.environ["AMPLIFIER_NO_PERSIST"] = "1"
        os.environ.pop("AMPLIFIER_STORAGE_DIR", None)
        
        manager = SessionManager()
        assert manager._store is None
        
        # Cleanup
        os.environ.pop("AMPLIFIER_NO_PERSIST", None)

    def test_provided_store_takes_precedence(self):
        """Test that explicitly provided store takes precedence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_store = SessionStore(storage_dir=Path(tmpdir))
            os.environ["AMPLIFIER_NO_PERSIST"] = "1"  # Should be ignored
            
            manager = SessionManager(store=custom_store)
            assert manager._store is custom_store
            assert str(manager._store.storage_dir) == tmpdir
            
            # Cleanup
            os.environ.pop("AMPLIFIER_NO_PERSIST", None)

    def test_list_saved_with_no_persist(self):
        """Test list_saved returns empty list when persistence disabled."""
        os.environ["AMPLIFIER_NO_PERSIST"] = "1"
        
        manager = SessionManager()
        saved = manager.list_saved()
        assert saved == []
        
        # Cleanup
        os.environ.pop("AMPLIFIER_NO_PERSIST", None)

    @pytest.mark.asyncio
    async def test_resume_fails_with_no_persist(self):
        """Test resume returns None when persistence disabled."""
        os.environ["AMPLIFIER_NO_PERSIST"] = "1"
        
        manager = SessionManager()
        session = await manager.resume("sess_does_not_exist")
        assert session is None
        
        # Cleanup
        os.environ.pop("AMPLIFIER_NO_PERSIST", None)
