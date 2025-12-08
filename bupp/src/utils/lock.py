import sqlite3
import time
import os
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, Any
import logging

class SQLiteLockManager:
    """
    SQLite-based distributed lock with automatic timeout and stale lock detection.
    
    Why SQLite is superior to file locks:
    1. ACID transactions - atomic read-modify-write operations
    2. Row-level locking - no manual file descriptor management
    3. Automatic cleanup - connection close releases locks
    4. Query support - can inspect lock state from any process
    5. Crash recovery - SQLite handles corrupted states
    """
    
    def __init__(self, db_path: Path | str = Path("browser_locks.db")):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize the locks table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS locks (
                    lock_name TEXT PRIMARY KEY,
                    holder_pid INTEGER NOT NULL,
                    acquired_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    holder_info TEXT
                )
            """)
            conn.commit()
    
    @contextmanager
    def acquire_lock(
        self, 
        lock_name: str, 
        timeout: int = 30,
        blocking_timeout: int = 10,
        holder_info: Optional[str] = None
    ):
        """
        Acquire a lock with automatic expiration and deadlock prevention.
        
        Args:
            lock_name: Unique identifier for the lock
            timeout: Lock auto-expires after this many seconds (prevents deadlock)
            blocking_timeout: How long to wait to acquire the lock
            holder_info: Optional metadata about lock holder
        
        Raises:
            TimeoutError: If lock cannot be acquired within blocking_timeout
        """
        pid = os.getpid()
        start_time = time.time()
        acquired = False
        
        try:
            # Try to acquire the lock
            while True:
                with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                    # Clean up expired locks first
                    self._cleanup_expired_locks(conn)
                    
                    # Try to acquire lock using INSERT OR REPLACE with transaction
                    current_time = time.time()
                    expires_at = current_time + timeout
                    
                    try:
                        # Check if lock exists and is valid
                        cursor = conn.execute(
                            "SELECT holder_pid, expires_at FROM locks WHERE lock_name = ?",
                            (lock_name,)
                        )
                        row = cursor.fetchone()
                        
                        if row is None:
                            # Lock doesn't exist, try to acquire it
                            conn.execute("""
                                INSERT INTO locks (lock_name, holder_pid, acquired_at, expires_at, holder_info)
                                VALUES (?, ?, ?, ?, ?)
                            """, (lock_name, pid, current_time, expires_at, holder_info))
                            conn.commit()
                            acquired = True
                            self.logger.info(f"Acquired lock '{lock_name}' (PID: {pid})")
                            break
                        
                        else:
                            holder_pid, lock_expires_at = row
                            
                            # Check if lock is expired
                            if current_time > lock_expires_at:
                                self.logger.warning(
                                    f"Lock '{lock_name}' expired (was held by PID {holder_pid}), acquiring"
                                )
                                # Lock expired, take it over
                                conn.execute("""
                                    UPDATE locks 
                                    SET holder_pid = ?, acquired_at = ?, expires_at = ?, holder_info = ?
                                    WHERE lock_name = ?
                                """, (pid, current_time, expires_at, holder_info, lock_name))
                                conn.commit()
                                acquired = True
                                break
                            
                            # Check if holder process is still alive (stale lock detection)
                            if not self._is_process_alive(holder_pid):
                                self.logger.warning(
                                    f"Lock '{lock_name}' held by dead process (PID {holder_pid}), acquiring"
                                )
                                conn.execute("""
                                    UPDATE locks 
                                    SET holder_pid = ?, acquired_at = ?, expires_at = ?, holder_info = ?
                                    WHERE lock_name = ?
                                """, (pid, current_time, expires_at, holder_info, lock_name))
                                conn.commit()
                                acquired = True
                                break
                    
                    except sqlite3.IntegrityError:
                        # Race condition: another process acquired it first
                        pass
                
                # Check timeout
                if time.time() - start_time > blocking_timeout:
                    lock_info = self.query_lock_status(lock_name)
                    raise TimeoutError(
                        f"Could not acquire lock '{lock_name}' within {blocking_timeout}s. "
                        f"Current holder: PID {lock_info.get('holder_pid')}"
                    )
                
                self.logger.debug(f"Waiting for lock '{lock_name}'...")
                time.sleep(0.1)
            
            yield
            
        finally:
            # Release lock if we acquired it
            if acquired:
                self._release_lock(lock_name, pid)
    
    def _release_lock(self, lock_name: str, pid: int):
        """Release a lock, but only if we're the holder."""
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                # Only delete if we're the current holder
                cursor = conn.execute(
                    "DELETE FROM locks WHERE lock_name = ? AND holder_pid = ?",
                    (lock_name, pid)
                )
                if cursor.rowcount > 0:
                    conn.commit()
                    self.logger.info(f"Released lock '{lock_name}' (PID: {pid})")
                else:
                    self.logger.warning(
                        f"Could not release lock '{lock_name}' - not the current holder"
                    )
        except Exception as e:
            self.logger.error(f"Error releasing lock '{lock_name}': {e}")
    
    def _cleanup_expired_locks(self, conn: sqlite3.Connection):
        """Remove locks that have expired."""
        current_time = time.time()
        cursor = conn.execute(
            "DELETE FROM locks WHERE expires_at < ?",
            (current_time,)
        )
        if cursor.rowcount > 0:
            conn.commit()
            self.logger.debug(f"Cleaned up {cursor.rowcount} expired locks")
    
    def _is_process_alive(self, pid: int) -> bool:
        """Check if a process is still running."""
        try:
            os.kill(pid, 0)  # Signal 0 checks existence without killing
            return True
        except (OSError, ProcessLookupError):
            return False
    
    def query_lock_status(self, lock_name: str) -> Dict[str, Any]:
        """
        Query the status of a lock - can be called from any process.
        
        Returns:
            dict with lock information including holder PID, timestamps, etc.
        """
        with sqlite3.connect(self.db_path, timeout=5.0) as conn:
            cursor = conn.execute("""
                SELECT holder_pid, acquired_at, expires_at, holder_info
                FROM locks 
                WHERE lock_name = ?
            """, (lock_name,))
            
            row = cursor.fetchone()
            
            if row is None:
                return {
                    "is_locked": False,
                    "lock_name": lock_name
                }
            
            holder_pid, acquired_at, expires_at, holder_info = row
            current_time = time.time()
            is_expired = current_time > expires_at
            is_alive = self._is_process_alive(holder_pid)
            
            return {
                "is_locked": not is_expired and is_alive,
                "lock_name": lock_name,
                "holder_pid": holder_pid,
                "holder_info": holder_info,
                "acquired_at": acquired_at,
                "expires_at": expires_at,
                "time_held_seconds": current_time - acquired_at,
                "time_until_expiry_seconds": max(0, expires_at - current_time),
                "is_expired": is_expired,
                "is_holder_alive": is_alive,
                "is_stale": not is_alive
            }
    
    def list_all_locks(self) -> list[Dict[str, Any]]:
        """List all current locks in the system."""
        with sqlite3.connect(self.db_path, timeout=5.0) as conn:
            cursor = conn.execute("""
                SELECT lock_name, holder_pid, acquired_at, expires_at, holder_info
                FROM locks
            """)
            
            locks = []
            current_time = time.time()
            
            for row in cursor.fetchall():
                lock_name, holder_pid, acquired_at, expires_at, holder_info = row
                is_expired = current_time > expires_at
                is_alive = self._is_process_alive(holder_pid)
                
                locks.append({
                    "lock_name": lock_name,
                    "holder_pid": holder_pid,
                    "holder_info": holder_info,
                    "acquired_at": acquired_at,
                    "expires_at": expires_at,
                    "is_expired": is_expired,
                    "is_holder_alive": is_alive,
                    "is_stale": not is_alive
                })
            
            return locks
    
    def force_release_lock(self, lock_name: str) -> bool:
        """
        Forcefully release a lock (admin operation).
        Use with caution - only for cleaning up truly stuck locks.
        """
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.execute(
                    "DELETE FROM locks WHERE lock_name = ?",
                    (lock_name,)
                )
                conn.commit()
                
                if cursor.rowcount > 0:
                    self.logger.warning(f"Forcefully released lock '{lock_name}'")
                    return True
                return False
        except Exception as e:
            self.logger.error(f"Error force-releasing lock '{lock_name}': {e}")
            return False