"""Secrets by name: the run config holds env-var *names*, never values.

A secret is read from the environment variable whose name the config gives
(``aug_key_env``, ``hf_token_env``), at the moment it is used. The value is
wrapped in :class:`Secret`, whose ``repr``/``str`` are redacted, so a stray
log line, exception or manifest cannot carry it. Error messages name the
variable, never its value.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

from jev_factory.cli._errors import EXIT_ENV_ERROR, CliError
from jev_factory.factory.config import RunConfig

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/pipeline-qwen.env.example",
    "commit": "9debdc6",
    "adaptations": [
        "HF_TOKEN_ENV/AUG_KEY_ENV: names are config keys; values are read only here",
        "values are wrapped so repr/str/format are redacted",
    ],
    "licence": "Apache-2.0",
}

REDACTED = "[REDACTED]"


class Secret:
    """A secret value that refuses to print itself."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        """The raw value, for the one call site that hands it to a client."""
        return self._value

    def __repr__(self) -> str:
        return f"Secret({REDACTED})"

    __str__ = __repr__

    def __format__(self, spec: str) -> str:
        return REDACTED

    def __bool__(self) -> bool:
        return bool(self._value)

    def __reduce__(self):  # refuse to pickle the value into a manifest/cache
        raise TypeError("Secret cannot be serialised")


def read_secret(
    config: RunConfig, name_key: str, *, environ: Mapping[str, str] | None = None
) -> Secret:
    """Read the secret held in the env var named by config key ``name_key``."""
    var = config.require(name_key)
    env = os.environ if environ is None else environ
    value = env.get(var, "")
    if not value:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=(
                f"secret not available: environment variable {var} "
                f"(named by {name_key}) is unset or empty"
            ),
            remediation=(
                f"inject {var} into the environment of this command; "
                "never put the value in config"
            ),
        )
    return Secret(value)


def scrub(text: str, secrets: Iterable[Secret]) -> str:
    """Replace any secret value occurring in ``text`` (defence in depth)."""
    for secret in secrets:
        raw = secret.reveal()
        if raw:
            text = text.replace(raw, REDACTED)
    return text
