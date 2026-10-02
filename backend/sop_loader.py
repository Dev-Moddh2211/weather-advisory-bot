from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


SUPPORTED_OPERATORS = {"equals", "==", ">", ">=", "<", "<=", "in", "membership", "between"}


class ConditionModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    field: str
    operator: Literal["equals", "==", ">", ">=", "<", "<=", "in", "membership", "between"]
    value: Any = None
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def validate_shape(self):
        if self.operator == "between" and (self.min is None or self.max is None):
            raise ValueError("between requires min and max")
        if self.operator != "between" and self.value is None:
            raise ValueError(f"{self.operator} requires value")
        return self


class ConditionsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    activity_any: list[str] | None = None
    intent_any: list[str] | None = None
    group_any: list[str] | None = None
    all: list[ConditionModel] | None = None
    any: list[ConditionModel] | None = None


class SOPModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    category: str = Field(default="Uncategorized", min_length=1)
    conditions: ConditionsModel
    severity: Literal["LOW", "MODERATE", "HIGH"]
    advice: str = Field(min_length=1)
    priority: int
    cite_as: str | None = None

    @model_validator(mode="after")
    def default_citation(self):
        if not self.cite_as:
            self.cite_as = f"{self.id} - {self.name}"
        return self


class SOPConfigModel(BaseModel):
    schema_version: int
    severity_order: dict[str, int]
    sops: list[SOPModel]

    @model_validator(mode="after")
    def validate_ids(self):
        ids = [sop.id for sop in self.sops]
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        if duplicates:
            raise ValueError(f"duplicate SOP IDs: {', '.join(duplicates)}")
        return self


def load_sops(path: str | Path = "sops/sops.yaml") -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    try:
        validated = SOPConfigModel.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid SOP configuration in {path}: {exc}") from exc
    return validated.model_dump(exclude_none=True)
