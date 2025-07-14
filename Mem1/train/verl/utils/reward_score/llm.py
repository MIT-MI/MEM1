import os
import torch
from typing import Dict, List, Union, Optional
from openai import OpenAI
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from litellm import completion
import os

os.environ['GEMINI_API_KEY'] = "AIzaSyA0ajJBdjEiv4Yz9UBTT-7bxe9KRVcMxCU"

class OpenAIClient():
    def __init__(self):
        load_dotenv()
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")

        self.client = OpenAI(api_key=openai_key)


    def generate_response(self, prompt, model="gpt-4o", temperature=0.01, force_json=False):
        try: 
            # Format messages properly with content type
            # messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            # if force_json:
            #     response = self.client.chat.completions.create(
            #         model=model,
            #         response_format={"type": "json_object"},
            #         temperature=temperature,
            #         messages=messages
            #     )
            # else:
            #     response = self.client.chat.completions.create(
            #         model=model,
            #         temperature=temperature,
            #         messages=messages
            #     )
            # return response.choices[0].message.content.strip()
            response = completion(
                model="gemini/gemini-2.0-flash", 
                messages=[{"role": "user", "content": prompt}]
            )
            return response['choices'][0]['message']['content'].strip()

        except Exception as e:
            return f"Error: {str(e)}"