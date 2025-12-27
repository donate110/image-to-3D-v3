from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

config_dir = Path(__file__).parent

class Settings(BaseSettings):
    api_title: str = "3D Generation pipeline Service"

    # API settings
    host: str = "0.0.0.0"
    port: int = 10006

    # GPU settings
    qwen_gpu: int = Field(default=0, env="QWEN_GPU")
    trellis_gpu: int = Field(default=0, env="TRELLIS_GPU")
    dtype: str = Field(default="bf16", env="QWEN_DTYPE")

    # Hugging Face settings
    hf_token: Optional[str] = Field(default=None, env="HF_TOKEN")

    # Generated files settings
    save_generated_files: bool = Field(default=False, env="SAVE_GENERATED_FILES")
    send_generated_files: bool = Field(default=False, env="SEND_GENERATED_FILES")
    output_dir: Path = Field(default=Path("generated_outputs"), env="OUTPUT_DIR")

    # Trellis settings
    trellis_model_id: str = Field(default="jetx/trellis-image-large", env="TRELLIS_MODEL_ID")
    trellis_sparse_structure_steps: int = Field(default=8, env="TRELLIS_SPARSE_STRUCTURE_STEPS")
    trellis_sparse_structure_cfg_strength: float = Field(default=5.75, env="TRELLIS_SPARSE_STRUCTURE_CFG_STRENGTH")
    trellis_slat_steps: int = Field(default=20, env="TRELLIS_SLAT_STEPS")
    trellis_slat_cfg_strength: float = Field(default=2.4, env="TRELLIS_SLAT_CFG_STRENGTH")
    trellis_num_oversamples: int = Field(default=3, env="TRELLIS_NUM_OVERSAMPLES")
    compression: bool = Field(default=False, env="COMPRESSION")

    # Qwen Edit settings
    qwen_edit_base_model_path: str = Field(default="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors",env="QWEN_EDIT_BASE_MODEL_PATH")
    qwen_edit_model_path: str = Field(default="Qwen/Qwen-Image-Edit-2509",env="QWEN_EDIT_MODEL_PATH")
    qwen_edit_height: int = Field(default=1024, env="QWEN_EDIT_HEIGHT")
    qwen_edit_width: int = Field(default=1024, env="QWEN_EDIT_WIDTH")
    num_inference_steps: int = Field(default=8, env="NUM_INFERENCE_STEPS")
    true_cfg_scale: float = Field(default=1.0, env="TRUE_CFG_SCALE")
    qwen_edit_prompt_path: Path = Field(default=config_dir.joinpath("qwen_edit_prompt.json"), env="QWEN_EDIT_PROMPT_PATH")

    # Background removal settings (Kamui - Enhanced)
    background_removal_model_id: str = Field(default="hiepnd11/rm_back2.0", env="BACKGROUND_REMOVAL_MODEL_ID")
    input_image_size: tuple[int, int] = Field(default=(1024, 1024), env="INPUT_IMAGE_SIZE") # (height, width)
    output_image_size: tuple[int, int] = Field(default=(518, 518), env="OUTPUT_IMAGE_SIZE") # (height, width)
    padding_percentage: float = Field(default=0.2, env="PADDING_PERCENTAGE")
    limit_padding: bool = Field(default=True, env="LIMIT_PADDING")
    
    # Kamui Enhancement: Mask Thresholding (Lowered for transparent objects)
    mask_threshold: float = Field(default=0.35, env="MASK_THRESHOLD", description="Threshold for object detection - lower captures transparent objects")
    mask_threshold_min: float = Field(default=0.3, env="MASK_THRESHOLD_MIN", description="Minimum allowed threshold")
    mask_threshold_max: float = Field(default=0.95, env="MASK_THRESHOLD_MAX", description="Maximum allowed threshold")
    use_adaptive_threshold: bool = Field(default=False, env="USE_ADAPTIVE_THRESHOLD", description="Use Otsu's method for threshold")
    
    # Kamui Enhancement: Mask Quality
    mask_quantization_bits: int = Field(default=0, env="MASK_QUANTIZATION_BITS", description="0=no quantization (best quality)")
    enable_antialiasing: bool = Field(default=True, env="ENABLE_ANTIALIASING", description="Smooth edges during resize")
    interpolation_mode: str = Field(default="bilinear", env="INTERPOLATION_MODE", description="Interpolation: nearest, bilinear, bicubic")
    
    # Kamui Enhancement: Smart Padding
    use_smart_padding: bool = Field(default=False, env="USE_SMART_PADDING", description="Adapt padding to object shape")
    adaptive_padding_factor: float = Field(default=1.5, env="ADAPTIVE_PADDING_FACTOR", description="Extra padding for elongated objects")
    
    # Kamui Enhancement: Object Validation (Relaxed for multi-object scenes)
    min_object_coverage: float = Field(default=0.02, env="MIN_OBJECT_COVERAGE", description="Minimum object size (2% of image) - lower for small objects")
    max_object_coverage: float = Field(default=0.98, env="MAX_OBJECT_COVERAGE", description="Maximum object size (98% of image)")
    
    # Kamui Enhancement: Quality Monitoring
    enable_quality_metrics: bool = Field(default=True, env="ENABLE_QUALITY_METRICS", description="Calculate quality metrics")
    log_centering_quality: bool = Field(default=False, env="LOG_CENTERING_QUALITY", description="Log object centering quality")
    
    # Kamui Enhancement: Transparency Enhancement for Glass Objects
    enable_transparency_boost: bool = Field(default=False, env="ENABLE_TRANSPARENCY_BOOST", description="Enhance transparency for glass/transparent objects")
    transparency_detection_threshold: float = Field(default=0.15, env="TRANSPARENCY_DETECTION_THRESHOLD", description="Variance threshold to detect transparent regions")
    transparency_boost_factor: float = Field(default=0.6, env="TRANSPARENCY_BOOST_FACTOR", description="How much to reduce alpha (0.6 = 60% reduction for glass)")
    edge_preserve_width: int = Field(default=10, env="EDGE_PRESERVE_WIDTH", description="Pixels from edge to preserve (not make transparent)")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

__all__ = ["Settings", "settings"]

