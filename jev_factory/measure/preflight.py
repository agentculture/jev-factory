"""Served-model preflight: refuse to measure a server that is not serving the model.

Before the first scored entry of a served run, ``GET <base_url>/models`` must
answer HTTP 200 with the model among the returned ids, and the served context
must equal the context the run is labelled with. A stopped or crashed server
otherwise turns every entry into a call error while the run still looks
plausible, and a server started at another context measures something else
than the report says.

vLLM reports ``max_model_len`` per model; llama-server (``owned_by`` is
``"llamacpp"``) does not, so its context is read from ``GET /props``
(``default_generation_settings.n_ctx``) at the server root. Only
``http://`` to this machine is accepted, a URL carrying credentials is
refused, and a redirect is never followed (it could leave localhost).
Stdlib only.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from urllib.parse import urlsplit

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/measure_skills.py",
    "commit": "9debdc6",
    "adaptations": [
        "require_localhost (lines 182-196), _NoRedirect (199-203), _llama_server_ctx"
        " (206-223) and preflight_models (226-285) kept; the scorer path only",
        "measure.py's preflight_models wrapper (lines 289-306) is folded in: failures raise"
        " PreflightError, which the measure run reports as an environment error (exit 2)",
        "GET /models is fetched without following redirects too (nvsh followed them there)",
    ],
    "licence": "Apache-2.0",
}

#: ``GET <base_url>/models`` timeout: short, since it only checks the server is up.
PREFLIGHT_TIMEOUT = 5.0
_HOST_ACCEPT = frozenset({"127.0.0.1", "::1", "localhost"})


class PreflightError(RuntimeError):
    """The server is not up, not serving the model, or serving another context."""


def require_localhost(base_url: str) -> None:
    """Raise ``ValueError`` unless *base_url* is ``http://`` to this machine, without userinfo."""
    parsed = urlsplit(base_url)
    if parsed.scheme != "http":
        raise ValueError(f"only http:// scheme accepted (got {parsed.scheme!r})")
    if parsed.username or parsed.password:
        raise ValueError("URL must not contain credentials (userinfo)")
    if parsed.hostname not in _HOST_ACCEPT:
        raise ValueError(f"host must be 127.0.0.1, ::1 or localhost (got {parsed.hostname!r})")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect: the answer must come from the localhost server asked."""

    def redirect_request(self, *_args, **_kwargs):  # noqa: D102
        return None


def llama_server_ctx(base_url: str, timeout: float) -> int | None:
    """llama-server's served context from ``GET /props`` at the server root, or ``None``."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    require_localhost(root)
    request = urllib.request.Request(root + "/props", method="GET")
    opener = urllib.request.build_opener(_NoRedirect)  # a redirect could leave localhost
    try:
        with opener.open(request, timeout=timeout) as response:  # nosec B310 - localhost only
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    settings = payload.get("default_generation_settings") if isinstance(payload, dict) else None
    n_ctx = settings.get("n_ctx") if isinstance(settings, dict) else None
    return n_ctx if isinstance(n_ctx, int) else None


def preflight_models(
    base_url: str,
    model: str,
    ctx: int | None = None,
    timeout: float = PREFLIGHT_TIMEOUT,
) -> None:
    """Raise :class:`PreflightError` unless *base_url* is up, serves *model*, at *ctx*."""
    try:
        require_localhost(base_url)
    except ValueError as exc:
        raise PreflightError(f"refusing to preflight {base_url}: {exc}") from exc
    url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(url, method="GET")
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:  # nosec B310 - localhost only
            status = response.status
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise PreflightError(f"{url} answered HTTP {exc.code}, not 200") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise PreflightError(f"cannot reach {url} to confirm the server is up: {exc}") from exc
    if status != 200:
        raise PreflightError(f"{url} answered HTTP {status}, not 200")
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise PreflightError(f"{url} did not answer valid JSON: {exc}") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    items = [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []
    ids = {item.get("id") for item in items}
    if model not in ids:
        raise PreflightError(
            f"{url} does not list {model!r} among its served models "
            f"({sorted(str(i) for i in ids if i)}); point --model at what the server is serving"
        )
    if ctx is None:
        return
    entry = next(item for item in items if item.get("id") == model)
    served = entry.get("max_model_len")
    if served is None and entry.get("owned_by") == "llamacpp":
        served = llama_server_ctx(base_url, timeout)
    if not isinstance(served, int):
        raise PreflightError(
            f"{url} does not report max_model_len (nor llama-server's /props n_ctx) for"
            f" {model!r}, so the served context cannot be checked against ctx={ctx}"
        )
    if served != ctx:
        raise PreflightError(
            f"{url} serves {model!r} with max_model_len={served}, but this run is labelled"
            f" ctx={ctx}; restart the server at context {ctx} (measure_ctx) or measure at"
            f" ctx={served}"
        )
