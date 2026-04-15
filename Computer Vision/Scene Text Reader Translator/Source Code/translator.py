"""Translation hook for Scene Text Reader Translator.

Provides a clean interface for optional text translation.
OCR is completely independent of translation — this module
is only invoked when ``translate_enabled=True`` in config.

No provider ships enabled by default. Custom providers can be injected by
subclassing :class:`TranslationProvider` and passing an instance to
:class:`Translator`.

Usage::

    from translator import Translator
    from config import SceneTextConfig

    cfg = SceneTextConfig(translate_enabled=True, translate_target_lang="es")
    t = Translator(cfg)
    translated = t.translate("Hello world")
    # Returns the original text unless a custom provider is injected.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

log = logging.getLogger("scene_text.translator")


class TranslationProvider(ABC):
    """Abstract base for translation providers."""

    @abstractmethod
    def translate(self, text: str, target_lang: str) -> str:
        """Translate *text* to *target_lang* and return the result."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""
        ...


class NoOpProvider(TranslationProvider):
    """Passthrough -- returns text unchanged.  Used when no provider is configured."""

    def translate(self, text: str, target_lang: str) -> str:
        return text

    def name(self) -> str:
        return "noop"


_BUILTIN_PROVIDERS: dict[str, type[TranslationProvider]] = {
    "noop": NoOpProvider,
}


class Translator:
    """Facade that delegates to a configured :class:`TranslationProvider`.

    If translation is disabled (``cfg.translate_enabled=False``), all calls
    return the original text unchanged.
    """

    def __init__(self, cfg, *, provider: TranslationProvider | None = None) -> None:
        self.cfg = cfg
        self._enabled = cfg.translate_enabled

        if provider is not None:
            self._provider = provider
        elif not self._enabled:
            self._provider = NoOpProvider()
        else:
            self._provider = self._resolve_provider(cfg.translate_provider)

    def translate(self, text: str) -> str:
        """Translate *text* to the configured target language.

        Returns the original text unchanged if translation is disabled
        or the provider is unavailable.
        """
        if not self._enabled or not text.strip():
            return text

        try:
            return self._provider.translate(
                text, self.cfg.translate_target_lang,
            )
        except Exception:
            log.warning(
                "Translation failed (provider=%s), returning original text",
                self._provider.name(),
                exc_info=True,
            )
            return text

    @property
    def provider_name(self) -> str:
        return self._provider.name()

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_provider(name: str) -> TranslationProvider:
        if not name:
            log.info("No translation provider configured -- using noop hook")
            return NoOpProvider()

        cls = _BUILTIN_PROVIDERS.get(name.lower())
        if cls is None:
            log.warning(
                "Unknown translation provider '%s' -- using noop hook. "
                "No bundled providers ship with this project.",
                name,
            )
            return NoOpProvider()

        try:
            instance = cls()
            log.info("Translation provider: %s", instance.name())
            return instance
        except Exception:
            log.warning(
                "Failed to initialise provider '%s' -- using noop",
                name,
                exc_info=True,
            )
            return NoOpProvider()
