from pydantic import BaseModel
from pathlib import Path


class SegmentorConfig(BaseModel):
    segmentor_path: Path
    segmentor_config: str
    detector_path: Path
    num_clusters: int
    num_kp: int
