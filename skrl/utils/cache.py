from typing import Any, Callable, Optional


class _Cache(object):
    def __init__(self) -> None:
        """General-purpose cache for objects."""
        self._cache = {}

    def get(self, obj: object, key: str, default: Optional[Callable[[], Any]] = None) -> Any:
        uid = id(obj)
        if uid not in self._cache:
            self._cache[uid] = {}
        if key not in self._cache[uid]:
            if default is None:
                return None
            self._cache[uid][key] = default()
        return self._cache[uid][key]

    def set(self, obj: object, key: str, value: Any) -> None:
        if id(obj) not in self._cache:
            self._cache[id(obj)] = {}
        self._cache[id(obj)][key] = value

    def clear(self) -> None:
        self._cache.clear()


cache = _Cache()
