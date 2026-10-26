"""Day 5: the judge seam. A local model that answers one question with typed JSON.

What it does
    1. Defines `Judge`: `ask(prompt, schema) -> schema instance`.
    2. Provides `make_ollama_judge`, a judge backed by a local Ollama model.
    3. Validates the raw reply with Pydantic in OUR code, retries once with the
       validation error in the prompt, then raises `JudgeError`.

Why it copies the triage pattern (app/triage.py `ask_validated`)
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

from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

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
    """

    def chat(model: str, prompt: str, json_schema: dict[str, Any]) -> str:
        """One Ollama call. Returns the raw reply text."""
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=model,
            temperature=0,
            format=json_schema,
            client_kwargs={"timeout": timeout_s},
        )
        content = llm.invoke(prompt).content
        return content if isinstance(content, str) else str(content)

    return chat


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
) -> Judge:
    """Return a judge for `model`. `chat` defaults to the real Ollama call.

    A timeout is not retried: a model that hung once will likely hang again,
    and a retry would double the wait.
    """
    return _OllamaJudge(model, chat or ollama_chat(timeout_s), retries)
