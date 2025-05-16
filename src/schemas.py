from pydantic import BaseModel
from pathlib import Path


class SegmentorConfig(BaseModel):
    detector_path: Path
    num_clusters: int
    num_kp: int
