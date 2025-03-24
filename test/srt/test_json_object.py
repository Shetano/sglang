"""
python3 -m unittest test_json_object.TestJSONObjectResponse.test_json_object_response
python3 -m unittest test_json_object.TestJSONObjectResponse.test_json_object_with_streaming
"""

import json
import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestJSONObjectResponse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--max-running-requests",
            "10",
            "--grammar-backend",
            "llguidance",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        cls.client = openai.Client(api_key="EMPTY", base_url=f"{cls.base_url}/v1")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_json_object_response(self):
        """Test that response_format json_object produces valid JSON."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {
                    "role": "user",
                    "content": "What is the capital of Bulgaria?"
                },
            ],
            temperature=0,
            max_tokens=128,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content

        # Verify the response is valid JSON
        try:
            js_obj = json.loads(text)
        except json.JSONDecodeError as e:
            self.fail(f"Response is not valid JSON. Error: {e}. Response: {text}")

        # Verify it's actually an object (dict)
        self.assertIsInstance(js_obj, dict, f"Response is not a JSON object: {text}")

    def test_json_object_with_streaming(self):
        """Test that streaming with json_object response format works correctly."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                # We are deliberately omitting "That produces JSON" from the system prompt to avoid misleading test results
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": "What is the capital of Bulgaria?"},
            ],
            temperature=0,
            max_tokens=128,
            response_format={"type": "json_object"},
            stream=True,
        )

        # Collect all chunks
        chunks = []
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                chunks.append(chunk.choices[0].delta.content)
        full_response = "".join(chunks)

        # Verify the combined response is valid JSON
        try:
            js_obj = json.loads(full_response)
        except json.JSONDecodeError as e:
            self.fail(f"Streamed response is not valid JSON. Error: {e}. Response: {full_response}")

        self.assertIsInstance(js_obj, dict)


if __name__ == "__main__":
    unittest.main()
