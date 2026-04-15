"""Vlm tools."""

import base64
import math
import time
from typing import List, Optional
from .base_tool import BaseTool

class VLMTool(BaseTool):
    """V l m tool."""
    
    def __init__(
        self,
        model: str = "gpt-oss-120b-ldm",
        vision_model: str = "qwen3vl-8b",
        api_key: str = "EMPTY",
        base_url: str = "http://stepcast-router.shai-core:9200/v1",
        max_retries: int = 20,
        timeout: int = 30
    ):
        """Init."""
        
        super().__init__(
            name="vlm_tool",
            description="Run multimodal reasoning with a VLM"
        )
        self.model = model
        self.vision_model = vision_model
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "VLMTool requires the `openai` package. Install project dependencies from `easy_r1/pyproject.toml`."
            ) from exc
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    @staticmethod
    def encode_image(image_path: str) -> str:
        """Encode image."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def execute(
        self,
        prompt: str = None,
        question: str = None,
        image_paths: Optional[List[str]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        use_vision: Optional[bool] = None,
        force_model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Execute."""

        text = prompt if prompt is not None else question
        if text is None:
            return "Error: No prompt or question provided"
        if image_paths is None:
            image_paths = []
        

        content = [{"type": "text", "text": text}]
        

        for path in image_paths:
            if image_paths is None:
                break
            try:
                if not path.startswith("http"):
                    base64_image = self.encode_image(path)
                    image_url = f"data:image/jpeg;base64,{base64_image}"
                else:
                    image_url = path
                
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            except Exception as e:
                print(f"Failed to process image {path}: {e}")
        
        messages = [{"role": "user", "content": content}]
        if isinstance(force_model, str) and force_model.strip():
            selected_model = force_model.strip()
        else:
            has_images = len(image_paths) > 0
            use_vision_mode = has_images if use_vision is None else bool(use_vision)
            selected_model = self.vision_model if use_vision_mode else self.model

        current_max_tokens = max(1, int(max_tokens))
        max_growth_tokens = max(current_max_tokens, min(8192, int(math.ceil(current_max_tokens * 8))))
        

        for attempt in range(self.max_retries):
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=messages,
                    model=selected_model,
                    temperature=temperature,
                    max_tokens=current_max_tokens,
                    timeout=self.timeout,
                )
                response = chat_completion.choices[0].message.content

                finish_reason = chat_completion.choices[0].finish_reason
                if finish_reason == "content_filter":
                    return "Error: Content was blocked by the safety policy."

                if response is None:
                    if finish_reason == "length" and current_max_tokens < max_growth_tokens and attempt < self.max_retries - 1:
                        next_max_tokens = min(max_growth_tokens, max(current_max_tokens * 2, current_max_tokens + 64))
                        print(
                            f"Warning: model returned None. Finish reason: length. "
                            f"Retrying with max_tokens {current_max_tokens} -> {next_max_tokens}."
                        )
                        current_max_tokens = next_max_tokens
                        continue
                    print(
                        f"Warning: model returned None. Finish reason: {finish_reason}. "
                        f"Current max_tokens={current_max_tokens}"
                    )
                    return "Error: model refused to answer (empty response)."

                if response:
                    return response.strip()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"Request failed, retrying in {wait_time}s... ({e})")
                    time.sleep(wait_time)
                else:
                    return f"Error: Request failed after {self.max_retries} attempts. {e}"
        
        return "Error: Empty Response"
    
    def answer_with_context(
        self,
        question: str,
        context: str,
        image_paths: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Answer with context."""
        full_question = f"Context:\n{context}\n\nQuestion: {question}"
        return self.execute(full_question, image_paths, **kwargs)
