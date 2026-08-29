from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..blocks import Block


class PdfBackend(ABC):
    """Renders an ordered list of ``blocks`` into a PDF.

    Returns the PDF as ``bytes`` when ``path`` is ``None``, otherwise
    writes it to ``path`` and returns ``None``. Both backends support
    both modes so callers never need to know which one is active.
    """

    @abstractmethod
    def render(self, blocks: List[Block], *, path: Optional[str] = None) -> Optional[bytes]:
        raise NotImplementedError
