from starlette.requests import Request
import ray
from ray import serve
from transformers import pipeline

# E.g. to check if Ray Serve is working, baseline.
@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1, "num_gpus": 0})
class Translator:
    def __init__(self):
        # Load model
        self.model = pipeline("translation_en_to_fr", model="t5-small")

    def translate(self, text: str) -> str:
        # Run inference
        model_output = self.model(text)

        # Post-process output to return only the translation text
        translation = model_output[0]["translation_text"]

        return translation

    async def __call__(self, http_request: Request) -> str:
        english_text: str = await http_request.json()
        return self.translate(english_text)

translator_app = Translator.bind()

# Use below to send POST request
# import requests
# english_text = "Hello world!"
# response = requests.post("http://127.0.0.1:8000/", json=english_text)
# french_text = response.text
# print(french_text)