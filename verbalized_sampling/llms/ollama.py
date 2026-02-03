# Copyright 2025 CHATS-Lab. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from typing import Any, Dict, List

from ollama import Client
from pydantic import BaseModel

from verbalized_sampling.llms.base import BaseLLM


class OllamaLLM(BaseLLM):
    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any],
        num_workers: int = 1,
        strict_json: bool = False,
    ):
        super().__init__(model_name, config, num_workers, strict_json)

        # Ollama requires a base_url
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.client = Client(host=base_url)

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        # Map OpenAI config params to Ollama options if needed
        options = {}
        if "temperature" in self.config:
            options["temperature"] = self.config["temperature"]
        if "top_p" in self.config:
            options["top_p"] = self.config["top_p"]
        if "seed" in self.config:
            options["seed"] = self.config["seed"]
        # Add other options as needed

        response = self.client.chat(
            model=self.model_name,
            messages=messages,
            options=options,
        )
        # response is a dict: {'model': '...', 'created_at': '...', 'message': {'role': 'assistant', 'content': '...'}}
        content = response["message"]["content"]

        if content:
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]
        return content

    def _parse_response_with_schema(self, response: str) -> List[Dict[str, Any]]:
        """Parse the response based on the provided schema."""
        try:
            if isinstance(response, str):
                # Clean up markdown code blocks if present
                clean_response = response.strip()
                if "```json" in clean_response:
                    start = clean_response.find("```json") + 7
                    end = clean_response.find("```", start)
                    if end != -1:
                        clean_response = clean_response[start:end].strip()
                elif "```" in clean_response:
                    start = clean_response.find("```") + 3
                    end = clean_response.rfind("```")
                    if end != -1 and end > start:
                        clean_response = clean_response[start:end].strip()

                parsed = json.loads(clean_response)

                # Handle double-escaped JSON strings (i.e., string inside a string)
                if isinstance(parsed, str):
                    parsed = json.loads(parsed)

                # Handle different schema types
                if "responses" in parsed:
                    # For schemas with a 'responses' field (SequenceResponse, StructuredResponseList, etc.)
                    responses = parsed["responses"]

                    if isinstance(responses, list):
                        result = []
                        for resp in responses:
                            if (
                                isinstance(resp, dict)
                                and "text" in resp
                                and any(
                                    key in resp
                                    for key in ["probability", "confidence", "perplexity", "nll"]
                                )
                            ):
                                # Combine probability/confidence/perplexity fields
                                if "probability" in resp:
                                    prob = resp["probability"]
                                elif "confidence" in resp:
                                    prob = resp["confidence"]
                                elif "perplexity" in resp:
                                    prob = resp["perplexity"]
                                elif "nll" in resp:
                                    prob = resp["nll"]
                                result.append({"response": resp["text"], "probability": prob})
                            elif isinstance(resp, dict) and "text" in resp:
                                # Response
                                result.append({"response": resp["text"], "probability": 1.0})
                            elif isinstance(resp, str):
                                # SequenceResponse (list of strings)
                                result.append({"response": resp, "probability": 1.0})
                        return result
                else:
                    # For direct response schemas (Response)
                    if "text" in parsed:
                        return [
                            {
                                "response": parsed["text"],
                                "probability": parsed.get("probability", 1.0),
                            }
                        ]
                    elif "response" in parsed:
                        return [
                            {
                                "response": parsed["response"],
                                "probability": parsed.get("probability", 1.0),
                            }
                        ]

                # Fallback: return the raw validated data
                return [{"response": str(parsed), "probability": 1.0}]

        except Exception as e:
            print(f"Error parsing response with schema: {e}")
            # If parsing fails, return a single response with probability 1.0
            return [{"response": response, "probability": 1.0}]

    def _chat_with_format(
        self, messages: List[Dict[str, str]], schema: BaseModel
    ) -> List[Dict[str, Any]]:
        try:
            # Map OpenAI config params to Ollama options
            options = {}
            if "temperature" in self.config:
                options["temperature"] = self.config["temperature"]
            if "top_p" in self.config:
                options["top_p"] = self.config["top_p"]
            if "seed" in self.config:
                options["seed"] = self.config["seed"]

            # Use schema directly if supported or fallback to 'json' and parsing
            # Ollama Python client supports `format=schema.model_json_schema()`
            schema_to_use = schema
            if hasattr(schema, "model_json_schema"):
                schema_to_use = schema.model_json_schema()
            elif (
                isinstance(schema, dict)
                and schema.get("type") == "json_schema"
                and "json_schema" in schema
            ):
                # Extract inner schema from OpenAI format
                schema_to_use = schema["json_schema"].get("schema", schema["json_schema"])

            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options=options,
                format=schema_to_use,
            )

            content = response["message"]["content"]
            parsed_response = self._parse_response_with_schema(content)
            return parsed_response
        except Exception as e:
            print(f"Error: {e}")
            return []
