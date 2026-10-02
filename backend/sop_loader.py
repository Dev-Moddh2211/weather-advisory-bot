from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


SUPPORTED_OPERATORS = {"equals", "==", ">", ">=", "<", "<=", "in", "membership", "between"}
SUPPORTED_CONDITION_FIELDS = {
    "activity",
    "intent",
    "user_group",
    "temperature_c",
    "wind_speed_kmh",
    "precipitation_mm",
    "precipitation_probability_pct",
    "uv_index",
    "weather_condition",
}


class ConditionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    operator: Literal["equals", "==", ">", ">=", "<", "<=", "in", "membership", "between"]
    value: Any = None
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def validate_shape(self):
        if self.field not in SUPPORTED_CONDITION_FIELDS:
            raise ValueError(f"unsupported condition field: {self.field}")
        if self.operator == "between" and (self.min is None or self.max is None):
            raise ValueError("between requires min and max")
        if self.operator != "between" and self.value is None:
            raise ValueError(f"{self.operator} requires value")
        return self


class FuzzyMembershipModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["ascending", "descending", "triangle", "trapezoid"]
    points: list[float] = Field(min_length=2, max_length=4)

    @model_validator(mode="after")
    def validate_points(self):
        expected = {"ascending": 2, "descending": 2, "triangle": 3, "trapezoid": 4}[self.kind]
        if len(self.points) != expected or any(left >= right for left, right in zip(self.points, self.points[1:])):
            raise ValueError(f"{self.kind} requires {expected} ordered points")
        return self


class FuzzySignalModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    membership: FuzzyMembershipModel
    weight: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_field(self):
        if self.field not in SUPPORTED_CONDITION_FIELDS:
            raise ValueError(f"unsupported fuzzy condition field: {self.field}")
        return self


class FuzzyConditionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    minimum_score: float = Field(ge=0, le=1)
    signals: list[FuzzySignalModel] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_weights(self):
        if sum(signal.weight for signal in self.signals) <= 0:
            raise ValueError("fuzzy signal weights must be positive")
        return self


class ConditionsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    activity_any: list[str] | None = None
    intent_any: list[str] | None = None
    group_any: list[str] | None = None
    all: list[ConditionModel] | None = None
    any: list[ConditionModel] | None = None
    fuzzy: FuzzyConditionModel | None = None

    @model_validator(mode="after")
    def require_condition(self):
        if not any(
            value is not None
            for value in (
                self.activity_any,
                self.intent_any,
                self.group_any,
                self.all,
                self.any,
                self.fuzzy,
            )
        ):
            raise ValueError("conditions must define at least one scope or rule")
        return self


class SOPModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    category: str = Field(default="Uncategorized", min_length=1)
    conditions: ConditionsModel
    severity: Literal["LOW", "MODERATE", "HIGH"]
    advice: str = Field(min_length=1)
    priority: int
    cite_as: str | None = None
    description: str | None = None
    policy_note: str | None = None

    @model_validator(mode="after")
    def default_citation(self):
        if not self.cite_as:
            self.cite_as = f"{self.id} - {self.name}"
        return self


class SOPConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int
    severity_order: dict[str, int]
    sops: list[SOPModel]

    @model_validator(mode="after")
    def validate_ids(self):
        ids = [sop.id for sop in self.sops]
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        if duplicates:
            raise ValueError(f"duplicate SOP IDs: {', '.join(duplicates)}")
        allowed_severities = {sop.severity for sop in self.sops}
        if set(self.severity_order) != {"LOW", "MODERATE", "HIGH"}:
            raise ValueError("severity_order must define LOW, MODERATE, and HIGH")
        if any(value <= 0 for value in self.severity_order.values()):
            raise ValueError("severity_order values must be positive")
        if len(set(self.severity_order.values())) != len(self.severity_order):
            raise ValueError("severity_order values must be unique")
        if not allowed_severities.issubset(self.severity_order):
            raise ValueError("every SOP severity must exist in severity_order")
        return self


DEFAULT_SOP_PATH = Path(__file__).resolve().parents[1] / "sops" / "sops.yaml"


def load_sops(path: str | Path = DEFAULT_SOP_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    try:
        validated = SOPConfigModel.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid SOP configuration in {path}: {exc}") from exc
    return validated.model_dump(exclude_none=True)
