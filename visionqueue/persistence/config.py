from dataclasses import dataclass

@dataclass
class PersistenceConfig:
    database_path: str = "data/visionqueue.db"
    enabled: bool = True
    measurement_interval_seconds: float = 5.0
    max_queue_size: int = 1000
