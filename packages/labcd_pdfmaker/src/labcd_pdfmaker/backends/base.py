from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..blocks import Block


class PdfBackend(ABC):
    # path=None -> returns bytes, path given -> writes there and returns None.
    # both backends honor this so callers don't care which one's active.

    @abstractmethod
    def render(self, blocks: List[Block], *, path: Optional[str] = None) -> Optional[bytes]:
        raise NotImplementedError
