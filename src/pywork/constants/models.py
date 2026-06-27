from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ModelProvider = Literal[
    "openai",
    "anthropic",
    "deepseek",
    "zhipu",
    "qwen",
    "minimax",
    "moonshot",
    "baidu",
    "hunyuan",
    "xunfei",
    "ollama_local",
    "ollama_cloud",
    "openai_compatible",
]

ApiFormat = Literal[
    "openai",
    "anthropic",
    "ollama",
]

ModelCapability = Literal[
    "chat",
    "tool_calling",
    "structured_outputs",
    "json_mode",
    "vision",
    "reasoning",
    "long_context",
    "fim",
    "web_search",
    "local",
    "cloud",
]

ModelStatus = Literal[
    "recommended",
    "available",
    "preview",
    "legacy",
]

ModelCostTier = Literal[
    "low",
    "medium",
    "high",
    "premium",
]


# ---------------------------------------------------------------------
# Default model
# ---------------------------------------------------------------------

DEFAULT_PROVIDER: ModelProvider = "deepseek"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_API_FORMAT: ApiFormat = "openai"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_API_KEY_ENV = "DEEPSEEK_API_KEY"

DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_MAX_RETRIES = 2
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 4096
DEFAULT_REASONING_EFFORT = "medium"


# ---------------------------------------------------------------------
# API key env names
# ---------------------------------------------------------------------

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"

DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
ZHIPU_API_KEY_ENV = "ZAI_API_KEY"
QWEN_API_KEY_ENV = "DASHSCOPE_API_KEY"
MINIMAX_API_KEY_ENV = "MINIMAX_API_KEY"
MOONSHOT_API_KEY_ENV = "MOONSHOT_API_KEY"
BAIDU_API_KEY_ENV = "BAIDU_API_KEY"
HUNYUAN_API_KEY_ENV = "HUNYUAN_API_KEY"
XUNFEI_API_KEY_ENV = "XUNFEI_API_KEY"

OLLAMA_API_KEY_ENV = "OLLAMA_API_KEY"


# ---------------------------------------------------------------------
# Base URLs
# ---------------------------------------------------------------------

OPENAI_BASE_URL = None
ANTHROPIC_BASE_URL = None

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MINIMAX_BASE_URL = "https://api.minimaxi.com/v1"
MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"
BAIDU_BASE_URL = "https://aistudio.baidu.com/llm/lmapi/v3"
HUNYUAN_BASE_URL = "https://api.hunyuan.cloud.tencent.com/v1"
XUNFEI_BASE_URL = "https://spark-api-open.xf-yun.com/v1"

OLLAMA_LOCAL_BASE_URL = "http://localhost:11434/api"
OLLAMA_LOCAL_OPENAI_BASE_URL = "http://localhost:11434/v1"

OLLAMA_CLOUD_BASE_URL = "https://ollama.com/api"
OLLAMA_CLOUD_OPENAI_BASE_URL = "https://ollama.com/v1"

OPENAI_COMPATIBLE_BASE_URL = None


# ---------------------------------------------------------------------
# Provider recommended aliases
# ---------------------------------------------------------------------

OPENAI_BEST_MODEL = "gpt-5.5"
OPENAI_FAST_MODEL = "gpt-5.4-mini"
OPENAI_CHEAP_MODEL = "gpt-5.4-nano"

ANTHROPIC_BEST_MODEL = "claude-fable-5"
ANTHROPIC_BALANCED_MODEL = "claude-opus-4-8"
ANTHROPIC_FAST_MODEL = "claude-sonnet-4-6"

DEEPSEEK_BEST_MODEL = "deepseek-v4-pro"
DEEPSEEK_FAST_MODEL = "deepseek-v4-flash"

ZHIPU_BEST_MODEL = "glm-5.2"
ZHIPU_BALANCED_MODEL = "glm-5.1"
ZHIPU_FAST_MODEL = "glm-4.5-air"

QWEN_BEST_MODEL = "qwen3.7-max"
QWEN_BALANCED_MODEL = "qwen3.6-plus"
QWEN_FAST_MODEL = "qwen-flash"

MINIMAX_BEST_MODEL = "MiniMax-M3"
MINIMAX_BALANCED_MODEL = "MiniMax-M2.7"
MINIMAX_FAST_MODEL = "MiniMax-M2.7-highspeed"

MOONSHOT_BEST_MODEL = "kimi-k2.7-code"
MOONSHOT_FAST_MODEL = "kimi-k2.7-code-highspeed"
MOONSHOT_GENERAL_MODEL = "kimi-k2.6"

BAIDU_BEST_MODEL = "ernie-5.1"
BAIDU_REASONING_MODEL = "ernie-5.0-thinking-preview"
BAIDU_FAST_MODEL = "ernie-4.5-turbo-128k-preview"

HUNYUAN_BEST_MODEL = "hunyuan-turbos-latest"

XUNFEI_BEST_MODEL = "4.0Ultra"
XUNFEI_BALANCED_MODEL = "generalv3.5"
XUNFEI_FAST_MODEL = "lite"

OLLAMA_LOCAL_DEFAULT_MODEL = "qwen3"
OLLAMA_CLOUD_DEFAULT_MODEL = "qwen3-coder:480b-cloud"


@dataclass(frozen=True)
class ProviderInfo:
    name: ModelProvider
    display_name: str
    default_model: str
    api_format: ApiFormat
    api_key_env: str | None
    base_url: str | None
    openai_compatible_base_url: str | None = None
    description: str = ""


@dataclass(frozen=True)
class ModelInfo:
    provider: ModelProvider
    name: str
    display_name: str
    capabilities: tuple[ModelCapability, ...]
    status: ModelStatus = "available"
    cost_tier: ModelCostTier = "medium"
    context_window: int | None = None
    max_output_tokens: int | None = None
    description: str = ""


PROVIDERS: dict[ModelProvider, ProviderInfo] = {
    "openai": ProviderInfo(
        name="openai",
        display_name="OpenAI",
        default_model=OPENAI_BEST_MODEL,
        api_format="openai",
        api_key_env=OPENAI_API_KEY_ENV,
        base_url=OPENAI_BASE_URL,
        description="Official OpenAI provider.",
    ),
    "anthropic": ProviderInfo(
        name="anthropic",
        display_name="Anthropic",
        default_model=ANTHROPIC_BALANCED_MODEL,
        api_format="anthropic",
        api_key_env=ANTHROPIC_API_KEY_ENV,
        base_url=ANTHROPIC_BASE_URL,
        description="Official Anthropic provider.",
    ),
    "deepseek": ProviderInfo(
        name="deepseek",
        display_name="DeepSeek",
        default_model=DEEPSEEK_FAST_MODEL,
        api_format="openai",
        api_key_env=DEEPSEEK_API_KEY_ENV,
        base_url=DEEPSEEK_BASE_URL,
        description="DeepSeek OpenAI-compatible provider.",
    ),
    "zhipu": ProviderInfo(
        name="zhipu",
        display_name="智谱 GLM",
        default_model=ZHIPU_BEST_MODEL,
        api_format="openai",
        api_key_env=ZHIPU_API_KEY_ENV,
        base_url=ZHIPU_BASE_URL,
        description="Zhipu / BigModel OpenAI-compatible provider.",
    ),
    "qwen": ProviderInfo(
        name="qwen",
        display_name="阿里千问 Qwen",
        default_model=QWEN_BEST_MODEL,
        api_format="openai",
        api_key_env=QWEN_API_KEY_ENV,
        base_url=QWEN_BASE_URL,
        description="Alibaba Cloud DashScope OpenAI-compatible provider.",
    ),
    "minimax": ProviderInfo(
        name="minimax",
        display_name="MiniMax",
        default_model=MINIMAX_BEST_MODEL,
        api_format="openai",
        api_key_env=MINIMAX_API_KEY_ENV,
        base_url=MINIMAX_BASE_URL,
        description="MiniMax OpenAI-compatible provider.",
    ),
    "moonshot": ProviderInfo(
        name="moonshot",
        display_name="Moonshot / Kimi",
        default_model=MOONSHOT_BEST_MODEL,
        api_format="openai",
        api_key_env=MOONSHOT_API_KEY_ENV,
        base_url=MOONSHOT_BASE_URL,
        description="Moonshot / Kimi OpenAI-compatible provider.",
    ),
    "baidu": ProviderInfo(
        name="baidu",
        display_name="百度文心 ERNIE",
        default_model=BAIDU_BEST_MODEL,
        api_format="openai",
        api_key_env=BAIDU_API_KEY_ENV,
        base_url=BAIDU_BASE_URL,
        description="Baidu ERNIE OpenAI-compatible provider.",
    ),
    "hunyuan": ProviderInfo(
        name="hunyuan",
        display_name="腾讯混元 Hunyuan",
        default_model=HUNYUAN_BEST_MODEL,
        api_format="openai",
        api_key_env=HUNYUAN_API_KEY_ENV,
        base_url=HUNYUAN_BASE_URL,
        description="Tencent Hunyuan OpenAI-compatible provider.",
    ),
    "xunfei": ProviderInfo(
        name="xunfei",
        display_name="讯飞星火 Spark",
        default_model=XUNFEI_BEST_MODEL,
        api_format="openai",
        api_key_env=XUNFEI_API_KEY_ENV,
        base_url=XUNFEI_BASE_URL,
        description="iFlytek Spark OpenAI-compatible provider.",
    ),
    "ollama_local": ProviderInfo(
        name="ollama_local",
        display_name="Ollama Local",
        default_model=OLLAMA_LOCAL_DEFAULT_MODEL,
        api_format="ollama",
        api_key_env=None,
        base_url=OLLAMA_LOCAL_BASE_URL,
        openai_compatible_base_url=OLLAMA_LOCAL_OPENAI_BASE_URL,
        description="Local Ollama runtime. Requires locally pulled models.",
    ),
    "ollama_cloud": ProviderInfo(
        name="ollama_cloud",
        display_name="Ollama Cloud",
        default_model=OLLAMA_CLOUD_DEFAULT_MODEL,
        api_format="ollama",
        api_key_env=OLLAMA_API_KEY_ENV,
        base_url=OLLAMA_CLOUD_BASE_URL,
        openai_compatible_base_url=OLLAMA_CLOUD_OPENAI_BASE_URL,
        description="Ollama hosted cloud models.",
    ),
    "openai_compatible": ProviderInfo(
        name="openai_compatible",
        display_name="Custom OpenAI Compatible",
        default_model="custom-model",
        api_format="openai",
        api_key_env=OPENAI_API_KEY_ENV,
        base_url=OPENAI_COMPATIBLE_BASE_URL,
        description="Generic OpenAI-compatible provider, for LM Studio, vLLM, SGLang, etc.",
    ),
}


MODEL_REGISTRY: dict[str, ModelInfo] = {
    # -----------------------------------------------------------------
    # OpenAI, max 3
    # -----------------------------------------------------------------
    "gpt-5.5": ModelInfo(
        provider="openai",
        name="gpt-5.5",
        display_name="GPT-5.5",
        status="recommended",
        cost_tier="premium",
        context_window=1_000_000,
        max_output_tokens=128_000,
        capabilities=(
            "chat",
            "tool_calling",
            "structured_outputs",
            "vision",
            "reasoning",
            "long_context",
        ),
        description="OpenAI flagship model for complex reasoning and coding.",
    ),
    "gpt-5.4-mini": ModelInfo(
        provider="openai",
        name="gpt-5.4-mini",
        display_name="GPT-5.4 Mini",
        status="available",
        cost_tier="medium",
        capabilities=(
            "chat",
            "tool_calling",
            "structured_outputs",
            "vision",
            "reasoning",
        ),
        description="OpenAI lower-cost, lower-latency model.",
    ),
    "gpt-5.4-nano": ModelInfo(
        provider="openai",
        name="gpt-5.4-nano",
        display_name="GPT-5.4 Nano",
        status="available",
        cost_tier="low",
        capabilities=(
            "chat",
            "tool_calling",
            "structured_outputs",
            "vision",
        ),
        description="OpenAI smallest fast model.",
    ),

    # -----------------------------------------------------------------
    # Anthropic, max 3
    # -----------------------------------------------------------------
    "claude-fable-5": ModelInfo(
        provider="anthropic",
        name="claude-fable-5",
        display_name="Claude Fable 5",
        status="recommended",
        cost_tier="premium",
        capabilities=(
            "chat",
            "tool_calling",
            "vision",
            "reasoning",
            "long_context",
        ),
        description="Anthropic highest-capability model candidate.",
    ),
    "claude-opus-4-8": ModelInfo(
        provider="anthropic",
        name="claude-opus-4-8",
        display_name="Claude Opus 4.8",
        status="recommended",
        cost_tier="premium",
        capabilities=(
            "chat",
            "tool_calling",
            "vision",
            "reasoning",
            "long_context",
        ),
        description="Anthropic Opus-tier model for complex reasoning and agentic coding.",
    ),
    "claude-sonnet-4-6": ModelInfo(
        provider="anthropic",
        name="claude-sonnet-4-6",
        display_name="Claude Sonnet 4.6",
        status="available",
        cost_tier="high",
        capabilities=(
            "chat",
            "tool_calling",
            "vision",
            "reasoning",
            "long_context",
        ),
        description="Anthropic balanced model for coding and agent workflows.",
    ),

    # -----------------------------------------------------------------
    # DeepSeek, max 2
    # -----------------------------------------------------------------
    "deepseek-v4-pro": ModelInfo(
        provider="deepseek",
        name="deepseek-v4-pro",
        display_name="DeepSeek V4 Pro",
        status="recommended",
        cost_tier="medium",
        context_window=1_000_000,
        max_output_tokens=384_000,
        capabilities=(
            "chat",
            "tool_calling",
            "json_mode",
            "reasoning",
            "long_context",
            "fim",
        ),
        description="DeepSeek stronger model for coding and agent workflows.",
    ),
    "deepseek-v4-flash": ModelInfo(
        provider="deepseek",
        name="deepseek-v4-flash",
        display_name="DeepSeek V4 Flash",
        status="recommended",
        cost_tier="low",
        context_window=1_000_000,
        max_output_tokens=384_000,
        capabilities=(
            "chat",
            "tool_calling",
            "json_mode",
            "reasoning",
            "long_context",
            "fim",
        ),
        description="DeepSeek default cost-effective model.",
    ),

    # -----------------------------------------------------------------
    # Zhipu GLM, max 3
    # -----------------------------------------------------------------
    "glm-5.2": ModelInfo(
        provider="zhipu",
        name="glm-5.2",
        display_name="GLM-5.2",
        status="recommended",
        cost_tier="high",
        context_window=1_000_000,
        max_output_tokens=128_000,
        capabilities=(
            "chat",
            "tool_calling",
            "structured_outputs",
            "json_mode",
            "reasoning",
            "long_context",
        ),
        description="Zhipu flagship GLM model.",
    ),
    "glm-5.1": ModelInfo(
        provider="zhipu",
        name="glm-5.1",
        display_name="GLM-5.1",
        status="available",
        cost_tier="high",
        context_window=200_000,
        max_output_tokens=128_000,
        capabilities=(
            "chat",
            "tool_calling",
            "structured_outputs",
            "reasoning",
            "long_context",
        ),
        description="Zhipu strong reasoning and coding model.",
    ),
    "glm-4.5-air": ModelInfo(
        provider="zhipu",
        name="glm-4.5-air",
        display_name="GLM-4.5-Air",
        status="available",
        cost_tier="low",
        context_window=128_000,
        max_output_tokens=96_000,
        capabilities=(
            "chat",
            "tool_calling",
            "reasoning",
        ),
        description="Zhipu lightweight model.",
    ),

    # -----------------------------------------------------------------
    # Alibaba Qwen, max 3
    # -----------------------------------------------------------------
    "qwen3.7-max": ModelInfo(
        provider="qwen",
        name="qwen3.7-max",
        display_name="Qwen3.7 Max",
        status="recommended",
        cost_tier="high",
        context_window=1_000_000,
        capabilities=(
            "chat",
            "tool_calling",
            "structured_outputs",
            "reasoning",
            "long_context",
        ),
        description="Qwen latest Max model for agentic coding, long-horizon reasoning, and complex software engineering.",
    ),
    "qwen3.6-plus": ModelInfo(
        provider="qwen",
        name="qwen3.6-plus",
        display_name="Qwen3.6 Plus",
        status="available",
        cost_tier="medium",
        context_window=1_000_000,
        capabilities=(
            "chat",
            "tool_calling",
            "structured_outputs",
            "reasoning",
            "long_context",
            "vision",
        ),
        description="Qwen balanced model for text, image, video, and general agent workloads.",
    ),
    "qwen-flash": ModelInfo(
        provider="qwen",
        name="qwen-flash",
        display_name="Qwen Flash",
        status="available",
        cost_tier="low",
        capabilities=(
            "chat",
            "tool_calling",
            "structured_outputs",
        ),
        description="Qwen fast and cost-effective model.",
    ),

    # -----------------------------------------------------------------
    # MiniMax, max 3
    # -----------------------------------------------------------------
    "MiniMax-M3": ModelInfo(
        provider="minimax",
        name="MiniMax-M3",
        display_name="MiniMax M3",
        status="recommended",
        cost_tier="high",
        context_window=1_000_000,
        capabilities=(
            "chat",
            "tool_calling",
            "structured_outputs",
            "vision",
            "reasoning",
            "long_context",
        ),
        description="MiniMax latest M-series model for agents, coding, tools, and long context.",
    ),
    "MiniMax-M2.7": ModelInfo(
        provider="minimax",
        name="MiniMax-M2.7",
        display_name="MiniMax M2.7",
        status="available",
        cost_tier="medium",
        context_window=204_800,
        capabilities=(
            "chat",
            "tool_calling",
            "structured_outputs",
            "reasoning",
            "long_context",
        ),
        description="MiniMax balanced reasoning model.",
    ),
    "MiniMax-M2.7-highspeed": ModelInfo(
        provider="minimax",
        name="MiniMax-M2.7-highspeed",
        display_name="MiniMax M2.7 Highspeed",
        status="available",
        cost_tier="medium",
        context_window=204_800,
        capabilities=(
            "chat",
            "tool_calling",
            "reasoning",
        ),
        description="MiniMax faster M2.7 model.",
    ),

    # -----------------------------------------------------------------
    # Moonshot / Kimi, max 3
    # -----------------------------------------------------------------
    "kimi-k2.7-code": ModelInfo(
        provider="moonshot",
        name="kimi-k2.7-code",
        display_name="Kimi K2.7 Code",
        status="recommended",
        cost_tier="high",
        context_window=256_000,
        capabilities=(
            "chat",
            "tool_calling",
            "vision",
            "reasoning",
            "long_context",
        ),
        description="Kimi strongest coding model.",
    ),
    "kimi-k2.7-code-highspeed": ModelInfo(
        provider="moonshot",
        name="kimi-k2.7-code-highspeed",
        display_name="Kimi K2.7 Code Highspeed",
        status="available",
        cost_tier="high",
        context_window=256_000,
        capabilities=(
            "chat",
            "tool_calling",
            "vision",
            "reasoning",
        ),
        description="Kimi faster coding model.",
    ),
    "kimi-k2.6": ModelInfo(
        provider="moonshot",
        name="kimi-k2.6",
        display_name="Kimi K2.6",
        status="available",
        cost_tier="medium",
        context_window=256_000,
        capabilities=(
            "chat",
            "tool_calling",
            "vision",
            "reasoning",
            "long_context",
        ),
        description="Kimi general agent and long-context model.",
    ),

    # -----------------------------------------------------------------
    # Baidu ERNIE, max 3
    # -----------------------------------------------------------------
    "ernie-5.1": ModelInfo(
        provider="baidu",
        name="ernie-5.1",
        display_name="ERNIE 5.1",
        status="recommended",
        cost_tier="high",
        context_window=128_000,
        max_output_tokens=65_536,
        capabilities=(
            "chat",
            "tool_calling",
            "structured_outputs",
            "reasoning",
            "long_context",
        ),
        description="Baidu ERNIE 5.1 model.",
    ),
    "ernie-5.0-thinking-preview": ModelInfo(
        provider="baidu",
        name="ernie-5.0-thinking-preview",
        display_name="ERNIE 5.0 Thinking Preview",
        status="preview",
        cost_tier="high",
        context_window=128_000,
        max_output_tokens=65_536,
        capabilities=(
            "chat",
            "reasoning",
            "long_context",
        ),
        description="Baidu ERNIE thinking preview model.",
    ),
    "ernie-4.5-turbo-128k-preview": ModelInfo(
        provider="baidu",
        name="ernie-4.5-turbo-128k-preview",
        display_name="ERNIE 4.5 Turbo 128K Preview",
        status="preview",
        cost_tier="medium",
        context_window=128_000,
        max_output_tokens=12_288,
        capabilities=(
            "chat",
            "tool_calling",
            "long_context",
        ),
        description="Baidu ERNIE 4.5 Turbo preview model.",
    ),

    # -----------------------------------------------------------------
    # Tencent Hunyuan, max 1 for now
    # -----------------------------------------------------------------
    "hunyuan-turbos-latest": ModelInfo(
        provider="hunyuan",
        name="hunyuan-turbos-latest",
        display_name="Hunyuan TurboS Latest",
        status="recommended",
        cost_tier="medium",
        capabilities=(
            "chat",
            "tool_calling",
            "reasoning",
            "web_search",
        ),
        description="Tencent Hunyuan TurboS model.",
    ),

    # -----------------------------------------------------------------
    # iFlytek Spark, max 3
    # -----------------------------------------------------------------
    "4.0Ultra": ModelInfo(
        provider="xunfei",
        name="4.0Ultra",
        display_name="Spark 4.0 Ultra",
        status="recommended",
        cost_tier="high",
        context_window=32_000,
        max_output_tokens=32_000,
        capabilities=(
            "chat",
            "tool_calling",
            "json_mode",
            "web_search",
        ),
        description="iFlytek Spark strongest non-thinking model.",
    ),
    "generalv3.5": ModelInfo(
        provider="xunfei",
        name="generalv3.5",
        display_name="Spark Max",
        status="available",
        cost_tier="medium",
        context_window=32_000,
        max_output_tokens=8_000,
        capabilities=(
            "chat",
            "tool_calling",
            "json_mode",
        ),
        description="iFlytek Spark Max model.",
    ),
    "lite": ModelInfo(
        provider="xunfei",
        name="lite",
        display_name="Spark Lite",
        status="available",
        cost_tier="low",
        context_window=8_000,
        max_output_tokens=4_000,
        capabilities=(
            "chat",
        ),
        description="iFlytek Spark lightweight model.",
    ),

    # -----------------------------------------------------------------
    # Ollama Local, max 3
    # -----------------------------------------------------------------
    "qwen3": ModelInfo(
        provider="ollama_local",
        name="qwen3",
        display_name="Qwen3 Local",
        status="recommended",
        cost_tier="low",
        capabilities=(
            "chat",
            "local",
        ),
        description="Local Ollama model. Requires: ollama pull qwen3",
    ),
    "llama3.2": ModelInfo(
        provider="ollama_local",
        name="llama3.2",
        display_name="Llama 3.2 Local",
        status="available",
        cost_tier="low",
        capabilities=(
            "chat",
            "local",
        ),
        description="Local Ollama model. Requires: ollama pull llama3.2",
    ),
    "deepseek-r1": ModelInfo(
        provider="ollama_local",
        name="deepseek-r1",
        display_name="DeepSeek R1 Local",
        status="available",
        cost_tier="low",
        capabilities=(
            "chat",
            "reasoning",
            "local",
        ),
        description="Local Ollama reasoning model. Requires: ollama pull deepseek-r1",
    ),

    # -----------------------------------------------------------------
    # Ollama Cloud, max 3
    # -----------------------------------------------------------------
    "qwen3-coder:480b-cloud": ModelInfo(
        provider="ollama_cloud",
        name="qwen3-coder:480b-cloud",
        display_name="Qwen3 Coder 480B Cloud",
        status="recommended",
        cost_tier="medium",
        capabilities=(
            "chat",
            "reasoning",
            "cloud",
        ),
        description="Ollama Cloud coding model.",
    ),
    "gpt-oss:120b-cloud": ModelInfo(
        provider="ollama_cloud",
        name="gpt-oss:120b-cloud",
        display_name="GPT OSS 120B Cloud",
        status="available",
        cost_tier="medium",
        capabilities=(
            "chat",
            "reasoning",
            "cloud",
        ),
        description="Ollama Cloud open model.",
    ),
    "deepseek-v3.1:671b-cloud": ModelInfo(
        provider="ollama_cloud",
        name="deepseek-v3.1:671b-cloud",
        display_name="DeepSeek V3.1 671B Cloud",
        status="available",
        cost_tier="medium",
        capabilities=(
            "chat",
            "reasoning",
            "cloud",
        ),
        description="Ollama Cloud DeepSeek model.",
    ),

    # -----------------------------------------------------------------
    # Generic OpenAI-compatible
    # -----------------------------------------------------------------
    "custom-model": ModelInfo(
        provider="openai_compatible",
        name="custom-model",
        display_name="Custom Model",
        status="available",
        cost_tier="medium",
        capabilities=(
            "chat",
        ),
        description="Placeholder model for custom OpenAI-compatible endpoints.",
    ),
}


def get_provider(provider: ModelProvider) -> ProviderInfo:
    return PROVIDERS[provider]


def get_default_provider() -> ProviderInfo:
    return get_provider(DEFAULT_PROVIDER)


def get_model_info(model_name: str) -> ModelInfo | None:
    return MODEL_REGISTRY.get(model_name)


def get_models_by_provider(provider: ModelProvider) -> list[ModelInfo]:
    return [
        model
        for model in MODEL_REGISTRY.values()
        if model.provider == provider
    ]


def get_recommended_models() -> list[ModelInfo]:
    return [
        model
        for model in MODEL_REGISTRY.values()
        if model.status == "recommended"
    ]


def get_available_models() -> list[ModelInfo]:
    return [
        model
        for model in MODEL_REGISTRY.values()
        if model.status in {"recommended", "available"}
    ]


def is_known_provider(provider: str) -> bool:
    return provider in PROVIDERS


def is_known_model(model_name: str) -> bool:
    return model_name in MODEL_REGISTRY


def model_supports(
    model_name: str,
    capability: ModelCapability,
) -> bool:
    model = get_model_info(model_name)

    if model is None:
        return False

    return capability in model.capabilities


def get_default_model_config() -> dict[str, object]:
    provider = get_default_provider()

    return {
        "provider": DEFAULT_PROVIDER,
        "model": DEFAULT_MODEL,
        "api_format": DEFAULT_API_FORMAT,
        "base_url": provider.base_url,
        "api_key_env": provider.api_key_env,
        "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
        "max_retries": DEFAULT_MAX_RETRIES,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "reasoning_effort": DEFAULT_REASONING_EFFORT,
    }


def get_provider_default_config(provider_name: ModelProvider) -> dict[str, object]:
    provider = get_provider(provider_name)

    return {
        "provider": provider.name,
        "model": provider.default_model,
        "api_format": provider.api_format,
        "base_url": provider.base_url,
        "api_key_env": provider.api_key_env,
        "openai_compatible_base_url": provider.openai_compatible_base_url,
    }


def list_known_providers() -> list[str]:
    return sorted(PROVIDERS.keys())


def list_known_models() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())


def print_provider_summary() -> None:
    print("Known providers:")
    for provider_name in list_known_providers():
        provider = PROVIDERS[provider_name]
        print(
            f"  - {provider.name:<18} "
            f"default={provider.default_model:<32} "
            f"format={provider.api_format:<8} "
            f"env={provider.api_key_env}"
        )


def print_model_summary() -> None:
    print("Recommended models:")
    for model in get_recommended_models():
        print(
            f"  - {model.name:<32} "
            f"[{model.provider}] "
            f"{model.display_name}"
        )

    print()
    print("All built-in models:")
    for model in MODEL_REGISTRY.values():
        print(
            f"  - {model.name:<32} "
            f"[{model.provider}] "
            f"status={model.status}"
        )


def main() -> int:
    print("Default model config:")
    for key, value in get_default_model_config().items():
        print(f"  {key}: {value}")

    print()
    print_provider_summary()

    print()
    print_model_summary()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())