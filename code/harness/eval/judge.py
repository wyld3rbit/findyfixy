"""Day 5: the judge seam. A local model that answers one question with typed JSON.

What it does
    1. Defines `Judge`: `ask(prompt, schema) -> schema instance`.
    2. Provides `make_ollama_judge`, a judge backed by a local Ollama model.
    3. Validates the raw reply with Pydantic in OUR code, retries once with the
       validation error in the prompt, then raises `JudgeError`.

Why it copies the validate-retry pattern of FindyFixy's triage (`ask_validated`)
    Constrained decoding guarantees shape, not meaning. The chat function
    returns raw text, so a bad reply fails here with a readable Pydantic
    message instead of inside a client library. That message is what the
    retry prompt carries. The retry template is local, not imported from
    triage: the triage one is worded for triage records, and bending it would
    couple two modules that change for different reasons.

Why a `ChatFn` is injected
    The real call (`ollama_chat`) is the only code that touches the network.
    Tests pass a fake, so no model or server is needed.

Failure modes (read before trusting a score)
    - Bad output (invalid JSON, wrong shape) is retried, then `JudgeError`.
    - A timeout or connection error from `chat` is also turned into
      `JudgeError`, and is not retried. For a batch evaluator one stuck call
      must cost one failed metric, not the whole run. This differs from
      triage, where an infrastructure fault is allowed to crash: there a
      human is waiting and a retry would hide an outage. Here the runner
      counts failures and reports them, so nothing is hidden.
    - Temperature 0 makes replies close to repeatable, not guaranteed:
      kernels and model reloads can still shift a token.
    - Worst case per call is 1 + retries model calls. At scale this is the
      cost lever: a judge metric costs several calls per sample.
"""

import json
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

T = TypeVar("T", bound=BaseModel)

# (model, prompt, json_schema) -> raw reply text. The schema is a plain dict so
# the seam carries no Pydantic types and a fake can ignore it.
ChatFn = Callable[[str, str, dict[str, Any]], str]

DEFAULT_TIMEOUT_S = 120.0

# Keep the echoed bad reply short. A runaway reply must not balloon the retry
# prompt, which costs input tokens.
_MAX_ECHO_CHARS = 500

RETRY_PROMPT = (
    "{prompt}\n\n"
    "Your previous reply was rejected.\n"
    "Previous reply: {reply}\n"
    "Validation error: {error}\n"
    "Reply again with one corrected JSON object."
)


class ChatText(str):
    """A judge reply's text that also carries what the server reported: tokens and model.

    It IS a `str`, so validation and every fake `ChatFn` that returns a plain
    string keep working. Tracing reads the extra fields; None means not
    reported, never zero.
    """

    input_tokens: int | None
    output_tokens: int | None
    served_by: str | None

    def __new__(cls, text: str, input_tokens: int | None = None, output_tokens: int | None = None,
                served_by: str | None = None) -> "ChatText":
        obj = super().__new__(cls, text)
        obj.input_tokens, obj.output_tokens, obj.served_by = input_tokens, output_tokens, served_by
        return obj


class JudgeError(Exception):
    """The judge gave no usable answer: bad output after retries, or a fault.

    Metrics catch this and report a failed result. They never turn it into a
    score.
    """


class Judge(Protocol):
    """Ask one question, get one validated object back.

    Raises `JudgeError` when no valid answer was obtained.
    """

    def ask(self, prompt: str, schema: type[T]) -> T:
        """Return an instance of `schema`, or raise `JudgeError`."""
        ...


def ollama_chat(timeout_s: float) -> ChatFn:
    """Build the real chat function. Imports lazily: no work at import time.

    Temperature 0 and the schema as `format` turn on greedy, grammar-constrained
    decoding. A timeout is set on the HTTP client. Needs Ollama running and the
    model pulled; otherwise the call raises, and the judge maps that to
    `JudgeError`.

    One client is built per (model, schema) and reused. A run makes about ten
    judge calls per case, and building a client each time sets up a fresh HTTP
    connection pool and throws away keep-alive. The schema is part of the key
    because Ollama takes it as a constructor argument. The cache is guarded by a
    lock because parallel workers share it; invoking a built client from
    several threads is safe.
    """
    clients: dict[tuple[str, str], Any] = {}
    guard = threading.Lock()

    def chat(model: str, prompt: str, json_schema: dict[str, Any]) -> str:
        """One Ollama call. Returns the raw reply text."""
        from langchain_ollama import ChatOllama

        key = (model, json.dumps(json_schema, sort_keys=True))
        with guard:
            llm = clients.get(key)
            if llm is None:
                llm = clients[key] = ChatOllama(
                    model=model,
                    temperature=0,
                    reasoning=False,
                    format=json_schema,
                    client_kwargs={"timeout": timeout_s},
                )
        message = llm.invoke(prompt)
        content = message.content if isinstance(message.content, str) else str(message.content)
        # Usage is optional: a client or server that reports none gives None, never 0.
        usage = getattr(message, "usage_metadata", None) or {}
        metadata = getattr(message, "response_metadata", None) or {}
        return ChatText(content, usage.get("input_tokens"), usage.get("output_tokens"), metadata.get("model"))

    return chat


def traced_chat(chat: ChatFn, tracer: "Tracer") -> ChatFn:
    """`chat`, with each call a generation span under the current span (the metric being judged).

    Every attempt is its own span, so a retry shows up as two calls. Tokens
    and the serving model come from a `ChatText` reply; a plain string reply
    (a fake) records none.
    """
    from harness.trace.spans import generation

    def call(model: str, prompt: str, json_schema: dict[str, Any]) -> str:
        """One traced judge call."""
        with generation(tracer, model, prompt, name=f"judge {model}") as gen:
            reply = chat(model, prompt, json_schema)
            gen.done(reply, input_tokens=getattr(reply, "input_tokens", None),
                     output_tokens=getattr(reply, "output_tokens", None),
                     response_model=getattr(reply, "served_by", None))
            return reply

    return call


class _OllamaJudge:
    """A `Judge` over an injected `ChatFn`. Holds config only, no connection."""

    def __init__(self, model: str, chat: ChatFn, retries: int) -> None:
        self._model = model
        self._chat = chat
        self._retries = retries

    def ask(self, prompt: str, schema: type[T]) -> T:
        """Call, validate, retry with the error text, then raise `JudgeError`."""
        json_schema = schema.model_json_schema()
        current = prompt
        last_error: Exception | None = None
        for _ in range(self._retries + 1):
            try:
                reply = self._chat(self._model, current, json_schema)
            except Exception as err:  # timeout, connection, 404: one outcome
                raise JudgeError(f"{self._model} call failed: {err!r}") from err
            try:
                return schema.model_validate_json(reply)
            except ValidationError as err:
                last_error = err
                current = RETRY_PROMPT.format(
                    prompt=prompt, reply=reply[:_MAX_ECHO_CHARS], error=err
                )
        raise JudgeError(
            f"{self._model} gave no valid {schema.__name__} after "
            f"{self._retries + 1} attempt(s): {last_error}"
        )


def make_ollama_judge(
    model: str,
    *,
    chat: ChatFn | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    retries: int = 1,
    tracer: "Tracer | None" = None,
) -> Judge:
    """Return a judge for `model`. `chat` defaults to the real Ollama call.

    A timeout is not retried: a model that hung once will likely hang again,
    and a retry would double the wait. With `tracer`, each call is a
    generation span (`traced_chat`).
    """
    base = chat or ollama_chat(timeout_s)
    return _OllamaJudge(model, traced_chat(base, tracer) if tracer is not None else base, retries)
