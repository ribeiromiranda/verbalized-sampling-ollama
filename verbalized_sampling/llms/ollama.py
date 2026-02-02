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

from openai import OpenAI
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

        # Ollama requires a base_url and a dummy api_key
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.client = OpenAI(base_url=base_url, api_key="ollama")

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.config,
        )
        response = response.choices[0].message.content
        if response:
            # response = response.replace("\n", "")
            # I think removing newlines might be aggressive for all models, but OpenAILLM does it.
            # I will follow OpenAILLM behavior for now.
            response = response.replace("\n", "")
            if response.startswith('"') and response.endswith('"'):
                response = response[1:-1]
        return response

    def _parse_response_with_schema(self, response: str) -> List[Dict[str, Any]]:
        """Parse the response based on the provided schema."""
        try:
            if isinstance(response, str):
                parsed = json.loads(response)

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
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **self.config,
                response_format=schema.model_json_schema() if hasattr(schema, 'model_json_schema') else schema,
            )

            response = completion.choices[0].message.content
            parsed_response = self._parse_response_with_schema(response)
            return parsed_response
        except Exception as e:
            print(f"Error: {e}")
            return []
