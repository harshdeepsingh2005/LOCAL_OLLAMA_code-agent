"""
File Lock Module

Provides file-level locking for safe concurrent access.
Ensures atomic operations and prevents race conditions.
"""

from __future__ import annotations

import fcntl
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional


class FileLockError(Exception):
    """Raised when file lock operations fail."""
    pass


class FileLockTimeout(FileLockError):
    """Raised when lock acquisition times out."""
    pass


@dataclass
class LockInfo:
    """Information about a held lock."""
    path: Path
    run_id: str
    acquired_at: datetime
    lock_type: str  # "read" or "write"
    fd: int


class FileLockManager:
    """
    Manages file locks for safe concurrent access.
    
    Provides:
    - Exclusive write locks
    - Shared read locks
    - Lock timeout handling
    - Per-run lock tracking
    """
    
    LOCK_EXTENSION = ".lock"
    DEFAULT_TIMEOUT = 30.0  # seconds
    
    def __init__(
        self,
        run_id: str,
        lock_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize the file lock manager.
        
        Args:
            run_id: Unique run identifier
            lock_dir: Directory for lock files (default: /tmp/agent_locks)
        """
        self._run_id = run_id
        self._lock_dir = lock_dir or Path("/tmp/agent_locks")
        self._lock_dir.mkdir(parents=True, exist_ok=True)
        
        # Track held locks
        self._held_locks: dict[str, LockInfo] = {}
    
    def _get_lock_path(self, file_path: Path) -> Path:
        """Get the lock file path for a given file."""
        # Create a safe lock filename from the path
        safe_name = str(file_path.resolve()).replace("/", "_").replace("\\", "_")
        return self._lock_dir / f"{safe_name}{self.LOCK_EXTENSION}"
    
    @contextmanager
    def write_lock(
        self,
        file_path: Path,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> Generator[None, None, None]:
        """
        Acquire an exclusive write lock on a file.
        
        Args:
            file_path: Path to the file to lock
            timeout: Maximum time to wait for lock
            
        Yields:
            None when lock is acquired
            
        Raises:
            FileLockTimeout: If lock cannot be acquired within timeout
        """
        lock_path = self._get_lock_path(file_path)
        lock_key = str(file_path.resolve())
        
        # Create lock file if it doesn't exist
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        
        start_time = time.monotonic()
        fd = None
        
        try:
            # Open lock file
            fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT)
            
            # Try to acquire exclusive lock
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except (IOError, OSError):
                    if time.monotonic() - start_time > timeout:
                        raise FileLockTimeout(
                            f"Timeout acquiring write lock for {file_path}"
                        )
                    time.sleep(0.1)
            
            # Write lock info
            lock_info = f"{self._run_id}\n{datetime.now(timezone.utc).isoformat()}\nwrite\n"
            os.write(fd, lock_info.encode())
            os.fsync(fd)
            
            # Track the lock
            self._held_locks[lock_key] = LockInfo(
                path=file_path,
                run_id=self._run_id,
                acquired_at=datetime.now(timezone.utc),
                lock_type="write",
                fd=fd,
            )
            
            yield
            
        finally:
            if fd is not None:
                # Release lock
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    os.close(fd)
                except (IOError, OSError):
                    pass
                
                # Remove from tracking
                self._held_locks.pop(lock_key, None)
                
                # Clean up lock file
                try:
                    lock_path.unlink(missing_ok=True)
                except (IOError, OSError):
                    pass
    
    @contextmanager
    def read_lock(
        self,
        file_path: Path,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> Generator[None, None, None]:
        """
        Acquire a shared read lock on a file.
        
        Args:
            file_path: Path to the file to lock
            timeout: Maximum time to wait for lock
            
        Yields:
            None when lock is acquired
        """
        lock_path = self._get_lock_path(file_path)
        lock_key = str(file_path.resolve())
        
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        
        start_time = time.monotonic()
        fd = None
        
        try:
            fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT)
            
            # Try to acquire shared lock
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    break
                except (IOError, OSError):
                    if time.monotonic() - start_time > timeout:
                        raise FileLockTimeout(
                            f"Timeout acquiring read lock for {file_path}"
                        )
                    time.sleep(0.1)
            
            self._held_locks[lock_key] = LockInfo(
                path=file_path,
                run_id=self._run_id,
                acquired_at=datetime.now(timezone.utc),
                lock_type="read",
                fd=fd,
            )
            
            yield
            
        finally:
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    os.close(fd)
                except (IOError, OSError):
                    pass
                
                self._held_locks.pop(lock_key, None)
    
    def is_locked(self, file_path: Path) -> bool:
        """Check if a file is currently locked."""
        lock_key = str(file_path.resolve())
        return lock_key in self._held_locks
    
    def get_held_locks(self) -> list[LockInfo]:
        """Get list of currently held locks."""
        return list(self._held_locks.values())
    
    def release_all(self) -> int:
        """
        Release all held locks.
        
        Returns:
            Number of locks released
        """
        released = 0
        for lock_key, info in list(self._held_locks.items()):
            try:
                fcntl.flock(info.fd, fcntl.LOCK_UN)
                os.close(info.fd)
                released += 1
            except (IOError, OSError):
                pass
            
            # Clean up lock file
            lock_path = self._get_lock_path(info.path)
            try:
                lock_path.unlink(missing_ok=True)
            except (IOError, OSError):
                pass
        
        self._held_locks.clear()
        return released
    
    def cleanup_stale_locks(self, max_age_seconds: float = 3600) -> int:
        """
        Clean up stale lock files.
        
        Args:
            max_age_seconds: Maximum age for lock files
            
        Returns:
            Number of stale locks cleaned
        """
        cleaned = 0
        now = time.time()
        
        for lock_file in self._lock_dir.glob(f"*{self.LOCK_EXTENSION}"):
            try:
                # Check file age
                stat = lock_file.stat()
                age = now - stat.st_mtime
                
                if age > max_age_seconds:
                    lock_file.unlink()
                    cleaned += 1
            except (IOError, OSError):
                pass
        
        return cleaned


class AtomicFileWriter:
    """
    Provides atomic file write operations.
    
    Writes to a temporary file and atomically renames
    to ensure no partial writes are visible.
    """
    
    def __init__(
        self,
        file_lock_manager: Optional[FileLockManager] = None,
    ) -> None:
        """
        Initialize the atomic writer.
        
        Args:
            file_lock_manager: Optional lock manager for coordination
        """
        self._lock_manager = file_lock_manager
    
    def write(
        self,
        path: Path,
        content: str,
        backup: bool = True,
    ) -> Optional[Path]:
        """
        Atomically write content to a file.
        
        Args:
            path: Target file path
            content: Content to write
            backup: Whether to create a backup
            
        Returns:
            Path to backup file if created, None otherwise
        """
        path = path.resolve()
        temp_path = path.with_suffix(path.suffix + ".tmp")
        backup_path = None
        
        try:
            # Create backup if file exists
            if backup and path.exists():
                backup_path = path.with_suffix(path.suffix + ".bak")
                import shutil
                shutil.copy2(path, backup_path)
            
            # Write to temp file
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_text(content, encoding="utf-8")
            
            # Ensure data is flushed to disk
            fd = os.open(str(temp_path), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            
            # Atomic rename
            os.rename(str(temp_path), str(path))
            
            return backup_path
            
        except Exception:
            # Clean up temp file on failure
            temp_path.unlink(missing_ok=True)
            raise
    
    def safe_delete(self, path: Path, backup: bool = True) -> Optional[Path]:
        """
        Safely delete a file with optional backup.
        
        Args:
            path: File to delete
            backup: Whether to create backup
            
        Returns:
            Path to backup file if created
        """
        path = path.resolve()
        backup_path = None
        
        if not path.exists():
            return None
        
        if backup:
            backup_path = path.with_suffix(path.suffix + ".deleted")
            import shutil
            shutil.copy2(path, backup_path)
        
        path.unlink()
        return backup_path
    
    def restore_backup(self, backup_path: Path) -> bool:
        """
        Restore a file from backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if restored successfully
        """
        if not backup_path.exists():
            return False
        
        # Determine original path
        original = backup_path.with_suffix("")
        if backup_path.suffix == ".bak":
            original = backup_path.with_suffix("")
        elif backup_path.suffix == ".deleted":
            original = backup_path.with_suffix("")
        
        import shutil
        shutil.copy2(backup_path, original)
        backup_path.unlink()
        
        return True
