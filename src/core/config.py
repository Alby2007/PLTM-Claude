"""Configuration management using Pydantic Settings"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DB_PATH: Path = PROJECT_ROOT / "data" / "memory.db"
    MODEL_CACHE: Path = Path.home() / ".cache" / "huggingface"

    # Models
    SMALL_MODEL: str = "Qwen/Qwen2.5-3B-Instruct"
    JUDGE_MODEL: str = "microsoft/Phi-3-mini-4k-instruct"

    # Thresholds
    PROMOTION_THRESHOLD: float = 0.85
    DISSOLUTION_FLOOR: float = 0.1

    # Tiered promotion configuration
    INSTANT_CONFIDENCE: float = 0.9
    FAST_HOURS: int = 4
    FAST_CONFIDENCE: float = 0.8
    STANDARD_HOURS: int = 12
    STANDARD_CONFIDENCE: float = 0.7
    SLOW_HOURS: int = 24

    # API keys (optional - for LLM tools)
    ANTHROPIC_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""
    ENABLE_API_FALLBACK: bool = False

    # Logging
    LOG_LEVEL: str = "INFO"

    # Strength initialization by provenance
    @property
    def STRENGTH_INIT(self) -> dict[str, float]:
        return {
            "user_stated": 0.8,
            "user_confirmed": 0.75,
            "inferred": 0.3,
            "corrected": 0.85,
        }


settings = Settings()
