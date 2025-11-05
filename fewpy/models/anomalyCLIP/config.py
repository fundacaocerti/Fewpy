from pydantic import BaseModel, Field


class AnomalyCLIPConfig(BaseModel):

    feature_list: list[int] = Field(
        default_factory=lambda: [6, 12, 18, 24],
        description="List of feature (scales) indices"
    )
    image_size: int = Field(default=700)
    depth: int = Field(default=9)
    n_ctx: int = Field(default=12)
    t_n_ctx: int = Field(default=4)

    kshot: int = Field(default=5)
    alpha: float = Field(1.0, description="Visual weight for normal prototype")
    beta: float = Field(1.0, description="Visual weight for anomalous prototype")
    scale_weights: list[float] = Field(
        default_factory=lambda: [0.5, 1.0, 2.0, 3.0],
        description="Weights for the scales"
    )
    obj_threshold: float = Field(0.1, description="Pixel intensity threshold for object detection")
    gamma: float = Field(2.0, description="Anomaly map intesity")
    contrast: float = Field(
        0.07, 
        description="Take control of sharpness between anomaly patches."
        "Larger values tend to give less self.config.contrast between anomalies"
    )

    user_prompts: list[str] | None = Field(
        None,
        description="List of prompts to guide detection"
    )
    softmax_temp: float = Field(
        0.07, 
        description="Softmax temperature for user text prompt attention. " \
        "Lower values pay more attention to visual anomalies described by the user prompt."
    )
    seed: int = Field(default=111)
    sigma: int = Field(default=4)
    cls_id: int | None = Field(default=None)