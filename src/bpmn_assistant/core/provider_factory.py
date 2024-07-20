from .anthropic_provider import AnthropicProvider
from .enums import Provider, OutputMode
from .llm_provider import LLMProvider
from .openai_provider import OpenAIProvider


class ProviderFactory:
    @staticmethod
    def get_provider(
        provider: Provider, api_key: str, output_mode: OutputMode = OutputMode.JSON
    ) -> LLMProvider:

        if provider == Provider.OPENAI:
            return OpenAIProvider(api_key, output_mode)
        elif provider == Provider.ANTHROPIC:
            return AnthropicProvider(api_key, output_mode)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
