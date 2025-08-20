import os
import requests
import logging
import torch
from typing import Dict, List, Union, Optional
from openai import OpenAI
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import litellm
from amem.memory_system import AgenticMemorySystem
import abc
from litellm import completion
import os


logger = logging.getLogger(__name__)

class BaseClient(abc.ABC):
    @abc.abstractmethod
    def generate_response(self, prompt, model="gpt-4o", temperature=0.01, force_json=False):
        pass

    def reset(self):
        pass
    
    @property
    def has_memory(self):
        return False


class VLLMOpenAIClient(BaseClient):
    def __init__(self):
        self.url = "http://localhost:8014"
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

    def generate_response(self, prompt, model="gpt-4o", temperature=0.01, force_json=False):
        try:
            response = requests.post(
                self.url + "/v1/chat/completions",
                json={
                    "model": model,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                    "stop": ["</search>", "</answer>"]
                }
            )

            choice = response.json()['choices'][0]

            content = choice["message"]["content"].strip()

            if choice["stop_reason"] == "</search>":
                content += "</search>"
            elif choice["stop_reason"] == "</answer>":
                content += "</answer>"

            return content

        except Exception as e:
            logger.error(f"Error: {str(e)}", exc_info=True)
            return f"Error: {str(e)}"

    def make_completion(self, initial_prompt, content, model="gpt-4o", temperature=0.01, force_json=False, is_last_turn=False):
        prompt_message = [{"role": "user", "content": initial_prompt}]
        prompt_message.append({"role": "assistant", "content": content})
        prompt_message = self.tokenizer.apply_chat_template(prompt_message, tokenize=False)

        # remove the <|im_end> at the end of the prompt
        prompt_message = prompt_message[:-len("<|im_end|>\n")]

        stop = []
        if is_last_turn:
            stop = ["</answer>"]
        else:
            stop = ["</search>", "</answer>"]

        try:
            response = requests.post(
                self.url + "/v1/completions",
                json={
                    "model": model,
                    "temperature": temperature,
                    "prompt": prompt_message,
                    "stop": stop,
                    "top_p": 0.95,
                    "top_k": -1,
                    "max_tokens": 1024,
                }
            )

            choice = response.json()['choices'][0]

            content = choice["text"].strip()

            if choice["stop_reason"] == "</search>":
                content += "</search>"
            elif choice["stop_reason"] == "</answer>":
                content += "</answer>"

            return content
        except Exception as e:
            logger.error(f"Error: {str(e)}", exc_info=True)
            return f"Error: {str(e)}" 

class LiteLLMClient(BaseClient):
    def __init__(self):
        litellm.drop_params = True
        assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY is not set"
        assert "OPENROUTER_API_KEY" in os.environ, "OPENROUTER_API_KEY is not set"

    def generate_response(self, prompt, model="openai/gpt-4o-mini", temperature=0.7, force_json=False):
        config = {
            "temperature": temperature,
            "top_p": 1,
            "provider": {
                "sort": "throughput"
            },
        }

        if model.startswith("openai/"):
            # drop provider
            config.pop("provider")

        if not model.startswith("openrouter/") and not model.startswith("openai/"):
            model = "openrouter/" + model

        try: 
            # Format messages properly with content type
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            if force_json:
                response = litellm.completion(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=messages,
                    **config
                )
            else:
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    **config
                )
            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error: {str(e)}"
    
    def make_completion(self, prompt, cur_obs, model="openai/gpt-4o-mini", temperature=0.01, force_json=False):
        config = {
            "temperature": temperature,
            "top_p": 1,
            "provider": {
                "sort": "throughput"
            },
        }

        if model.startswith("openai/"):
            # drop provider
            config.pop("provider")

        if not model.startswith("openrouter/") and not model.startswith("openai/"):
            model = "openrouter/" + model

        try: 
            # Format messages properly with content type
            messages = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": cur_obs}]}
            ]
            if force_json:
                response = litellm.completion(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=messages,
                    **config
                )
            else:
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    **config
                )
            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error: {str(e)}"


class AMemClient(BaseClient):
    def __init__(self):
        assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY is not set"
        assert "OPENROUTER_API_KEY" in os.environ, "OPENROUTER_API_KEY is not set"
        litellm.drop_params = True
        # Initialize the memory system 🚀
        self.memory_system = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',  # Embedding model for ChromaDB
            llm_backend="openai",           # LLM backend (openai/ollama)
            llm_model="gpt-4o-mini"         # LLM model name
        )
        self.memories = []


    def chat_with_memories(self, message: str, model: str, temperature: float = 0.01, force_json: bool = False, user_id: str = "default_user") -> str:
        # Retrieve relevant memories
        relevant_memories = self.memory_system.search_agentic(message, k=3)
        memories_str = "\n".join(f"- {entry['content']}" for entry in relevant_memories)
        self.memories.append(memories_str)
        # Generate Assistant response
        system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\nUser Memories:\n{memories_str}"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]

        config = {
            "temperature": temperature, 
            "top_p": 0.95,
            "provider": {
                "sort": "throughput"
            }
        }
        if force_json:
            config["response_format"] = {"type": "json_object"}
        
        if model.startswith("openai/"):
            config.pop("provider")
        
        if not model.startswith("openrouter/") and not model.startswith("openai/"):
            model = "openrouter/" + model
        
        response = litellm.completion(model=model, messages=messages, **config)
        assistant_response = response.choices[0].message.content.strip()

        return assistant_response


    def generate_response(self, prompt, model="openai/gpt-4o-mini", temperature=0.01, force_json=False):
        return self.chat_with_memories(prompt, model=model, temperature=temperature, force_json=force_json)


    @property
    def has_memory(self):
        return True
    
    def reset(self):
        self.memory_system = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',  # Embedding model for ChromaDB
            llm_backend="openai",           # LLM backend (openai/ollama)
            llm_model="gpt-4o-mini"         # LLM model name
        )
        self.memories = []