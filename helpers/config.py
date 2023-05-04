from typing import Literal
from pathlib import Path
import json

import yaml
from pydantic import BaseModel, conlist, validator


class Data(BaseModel):
    train_data_path: Path
    test_data_path: Path

    split: conlist(float, min_items=3, max_items=3)

    @validator('train_data_path', 'test_data_path')
    def convert_to_path(cls, value):
        path = Path(value)
        assert path.exists()
        return path

class Training(BaseModel):
    project_name: str
    experiment_name: str
    description: str
    batch_size: int
    epochs: int
    precision: Literal[16, 32]
    seed: int

class Model(BaseModel):
    model_name: str
    unfreeze_layers: list[str]

class Config(BaseModel):
    data: Data
    model: Model
    training: Training
    raw_data: dict

    @classmethod
    def from_yaml_file(cls, file_path: Path):
        with open(file_path) as file:
            data = yaml.safe_load(file)
        return cls(raw_data=data, **data)
    
    def to_dict(self) -> dict:
        return self.raw_data
    
    def __repr__(self):
        return json.dumps(self.raw_data, indent=4)
