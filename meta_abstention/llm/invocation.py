import time
from meta_abstention.config import llm
from typing import List
import json
import hashlib

class Prompt:
    class Message:
        def __init__(self, role: str, content: str):
            self.role = role
            self.content = content

        @staticmethod
        def load_from_json(j):
            return Prompt.Message(j['role'], j['content'])

    def __init__(self, messages: List[Message], temp: float = llm['default-temp'],
                 sample_size: int = llm['default-sample-size'],
                 logprobs: bool = False):
        self.messages = messages
        self.temp = temp
        self.sample_size = sample_size
        self.logprobs = logprobs
        self.model = None

    def hash(self):
        if self.model is None:
            raise ValueError("Model is required to hash the prompt")

        return hashlib.md5(str(json.dumps(self, default=lambda o: o.__dict__)).encode('utf-8')).hexdigest()

    @staticmethod
    def load_from_json(j):
        return Prompt([Prompt.Message.load_from_json(m) for m in j['messages']],
                      j['temp'],
                      j['sample_size'],
                      j.get('logprobs', False))

class Response:
    class Sample:
        def __init__(self, content: str,
                     token_logprobs: List[float] | None = None,
                     tokens: List[str] | None = None):
            self.content = content
            self.token_logprobs = token_logprobs
            self.tokens = tokens

        @staticmethod
        def load_from_json(j):
            return Response.Sample(j['content'],
                                   j.get('token_logprobs'),
                                   j.get('tokens'))

    def __init__(self, samples: List[Sample]):
        self.samples = samples

    @property
    def first_content(self) -> str:
        if not self.samples:
            raise ValueError("Response has no samples")
        return self.samples[0].content

    @property
    def first_sample(self) -> Sample:
        if not self.samples:
            raise ValueError("Response has no samples")
        return self.samples[0]

    @staticmethod
    def load_from_json(j):
        return Response([Response.Sample.load_from_json(s) for s in j['samples']])

class Invocation:
    def __init__(self, prompt: Prompt, response: Response, current_time: float = time.time()):
        self.prompt = prompt
        self.response = response
        self.invocation_time = current_time

    @staticmethod
    def load_from_json(j):
        return Invocation(prompt=Prompt.load_from_json(j['prompt']),
                          response=Response.load_from_json(j['response']),
                          current_time=j['invocation_time'])