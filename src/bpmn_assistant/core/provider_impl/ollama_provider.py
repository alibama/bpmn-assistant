import json
from typing import Generator, Any
from ollama import chat
from bpmn_assistant.core.enums import OutputMode, MessageRole
from bpmn_assistant.core.llm_provider import LLMProvider
from bpmn_assistant.config import logger

class OllamaProvider(LLMProvider):
    def __init__(self, output_mode: OutputMode = OutputMode.JSON):
        self.output_mode = output_mode

    def call(
        self,
        model: str,
        prompt: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str | dict[str, Any]:
        """
        Implementation of the Ollama API call.
        """
        # Convert messages to Ollama format
        ollama_messages = [
            {"role": msg["role"], "content": msg["content"]} for msg in messages
        ]
        ollama_messages.append({"role": "user", "content": prompt})

        response = chat(
            model=model,
            messages=ollama_messages,
            options={
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        )

        raw_output = response["message"]["content"]

        if not raw_output:
            raise ValueError("Empty response from Ollama")

        return self._process_response(raw_output)

    def stream(
        self,
        model: str,
        prompt: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> Generator[str, None, None]:
        """
        Implementation of the Ollama API stream.
        """
        ollama_messages = [
            {"role": msg["role"], "content": msg["content"]} for msg in messages
        ]
        ollama_messages.append({"role": "user", "content": prompt})

        response = chat(
            model=model,
            messages=ollama_messages,
            options={
                "num_predict": max_tokens,
                "temperature": temperature,
            },
            stream=True
        )

        for chunk in response:
            if chunk.get("message", {}).get("content"):
                yield chunk["message"]["content"]

    def get_initial_messages(self) -> list[dict[str, str]]:
        return (
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to output JSON.",
                }
            ]
            if self.output_mode == OutputMode.JSON
            else []
        )

    def add_message(
        self, messages: list[dict[str, str]], role: MessageRole, content: str
    ) -> None:
        message_role = "assistant" if role == MessageRole.ASSISTANT else "user"
        messages.append({"role": message_role, "content": content})

    def check_model_compatibility(self, model: str) -> bool:
        # You might want to implement a proper model check here
        # For now, we'll accept any model string as Ollama is more flexible
        return True

    def _process_response(self, raw_output: str) -> str | dict[str, Any]:
        """
        Process the raw output from the model. Returns the appropriate response based on the output mode.
        If the output mode is JSON, the raw output is parsed and returned as a dict.
        If the output mode is text, the raw output is returned as is.
        """
        if self.output_mode == OutputMode.JSON:
            try:
                result = json.loads(raw_output)

                if not isinstance(result, dict):
                    raise ValueError(f"Invalid JSON response from Ollama: {result}")

                return result
            except json.decoder.JSONDecodeError as e:
                logger.error(f"JSONDecodeError: {e}")
                logger.error(f"Raw output: {raw_output}")
                raise Exception("Invalid JSON response from Ollama") from e
        elif self.output_mode == OutputMode.TEXT:
            return raw_output
        else:
            raise ValueError(f"Unsupported output mode: {self.output_mode}")
