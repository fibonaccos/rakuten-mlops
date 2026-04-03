from datetime import datetime, timezone

from pydantic import BaseModel, Field


class PredictionInputOptions(BaseModel):
    """
    Input options schema for predictions.
    """

    return_confidence: bool = Field(
        default=False, description="Get the probability of the label predicted."
    )
    return_distribution: bool = Field(
        default=False, description="Get the probability distribution of all labels."
    )


class BasePredictionInput(BaseModel):
    """
    Base input fields for predictions.
    """

    designation: str = Field(
        ..., description="Main informations about the product (e.g. title, name, ...)."
    )
    description: str | None = Field(
        None, description="Optional description of the product (e.g. details, ...)."
    )


class SinglePredictionInput(BaseModel):
    """
    Schema for single input predictions.
    """

    inputs: BasePredictionInput
    options: PredictionInputOptions | None = None


class BatchPredictionInput(BaseModel):
    """
    Schema for batch inputs predictions.
    """

    inputs: list[BasePredictionInput]
    options: PredictionInputOptions | None = None


class BasePredictionOutput(BaseModel):
    """
    Base output fields for predictions.
    """

    label: str = Field(..., description="Predicted label.")
    confidence: float | None = Field(
        None, description="Probability of the predicted label."
    )
    distribution: dict[str, float] | None = Field(
        None, description="Probability distribution of the labels."
    )


class PredictionOutputMetadata(BaseModel):
    """
    Metadata fields for predictions.
    """

    model_name: str = Field(..., description="Model used for the prediction.")
    model_version: str | None = Field(None, description="Model version.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of the prediction request.",
    )
    inference_time_ms: float = Field(
        ..., description="Time of inference of the prediction in milliseconds."
    )


class SinglePredictionOutput(BaseModel):
    """
    Schema for single output predictions.
    """

    results: BasePredictionOutput
    metadata: PredictionOutputMetadata


class BatchPredictionOutput(BaseModel):
    """
    Schema for batch outputs predictions.
    """

    results: list[BasePredictionOutput]
    metadata: PredictionOutputMetadata
