import queue
import threading
import time
import logging
import sqlite3
from typing import Optional, Dict, Any
from .models import MeasurementRecord, Ack

logger = logging.getLogger(__name__)

class PersistenceWriter:
    def __init__(self, repository, max_queue_size: int = 1000):
        self._repo = repository
        self._critical_queue = queue.Queue(maxsize=max_queue_size)
        self._ack_queue = queue.Queue()
        self._pending_measurement: Optional[MeasurementRecord] = None
        self._measurement_lock = threading.Lock()
        self._write_lock = threading.Lock()
        
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        self._stats = {
            "total_written": 0,
            "total_errors": 0,
            "enqueue_failures": 0,
            "measurements_coalesced": 0
        }
        self._is_healthy = True
        
    @property
    def stats(self) -> Dict[str, int]:
        st = self._stats.copy()
        st["queue_size"] = self._critical_queue.qsize()
        st["is_healthy"] = int(self._is_healthy)
        return st

    def start_async_worker(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._writer_loop, name="PersistenceDBWorker", daemon=False)
            self._thread.start()
            
    def stop_async_worker(self, timeout: float = 5.0):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            
    def enqueue_critical(self, record) -> None:
        # Raises queue.Full if full
        self._critical_queue.put_nowait(record)
        
    def enqueue_measurement(self, record: MeasurementRecord) -> None:
        with self._measurement_lock:
            if self._pending_measurement is not None:
                self._stats["measurements_coalesced"] += 1
            self._pending_measurement = record
            
    def get_ack_nowait(self) -> Optional[Ack]:
        try:
            return self._ack_queue.get_nowait()
        except queue.Empty:
            return None
            
    def flush_all(self, timeout: float = 5.0):
        # Move pending measurement to a synchronous write if needed, or wait for queues to drain.
        start = time.time()
        while not self._critical_queue.empty():
            if time.time() - start > timeout:
                logger.warning("Flush timeout waiting for critical queue to drain")
                break
            time.sleep(0.05)
            
        with self._measurement_lock:
            m = self._pending_measurement
            self._pending_measurement = None
        if m:
            self.write_sync(m)
            
    def write_sync(self, record, max_retries: int = 3) -> bool:
        return self._write_with_retry(record, max_retries)

    def _writer_loop(self):
        while not self._stop_event.is_set() or not self._critical_queue.empty():
            record = None
            is_critical = False
            
            try:
                record = self._critical_queue.get_nowait()
                is_critical = True
            except queue.Empty:
                with self._measurement_lock:
                    if self._pending_measurement:
                        record = self._pending_measurement
                        self._pending_measurement = None
                        
            if record is None:
                if self._stop_event.is_set() and self._critical_queue.empty():
                    break
                time.sleep(0.01)
                continue
                
            success = self._write_with_retry(record)
            
            if is_critical:
                self._ack_queue.put(Ack(record=record, success=success))
                self._critical_queue.task_done()
                
    def _write_with_retry(self, record, max_retries: int = 3) -> bool:
        for attempt in range(max_retries + 1):
            try:
                with self._write_lock:
                    self._repo.execute(record)
                self._stats["total_written"] += 1
                self._is_healthy = True
                return True
            except sqlite3.OperationalError as e:
                if attempt < max_retries:
                    time.sleep(0.1 * (2 ** attempt))
                    continue
                self._stats["total_errors"] += 1
                self._is_healthy = False
                logger.error("Final SQLite failure for %s after %d retries: %s", type(record).__name__, max_retries, e)
                return False
            except Exception as e:
                self._stats["total_errors"] += 1
                self._is_healthy = False
                logger.error("Non-retryable SQLite failure for %s: %s", type(record).__name__, e)
                return False
        return False
