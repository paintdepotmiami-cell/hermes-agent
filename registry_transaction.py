"""Host-private copy-on-write primitives for process-local registries.

The plugin manager uses these objects only for short prepare/validate/install
cycles.  Arbitrary provider and plugin callbacks run while populating isolated
views, never while a live registry lock is held.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Iterable, Optional


class RegistryTransactionConflict(RuntimeError):
    """A live registry changed after a transaction captured its baseline."""

    def __init__(
        self,
        surface: str,
        expected_generation: int,
        actual_generation: int,
    ) -> None:
        self.surface = surface
        self.expected_generation = expected_generation
        self.actual_generation = actual_generation
        super().__init__(
            f"{surface} registry changed during plugin registration "
            f"(generation {expected_generation} -> {actual_generation}); "
            "refusing commit"
        )


class _MappingSnapshot:
    __slots__ = ("_owner", "_items", "_generation")

    def __init__(
        self,
        owner: "MappingRegistry",
        items: tuple[tuple[str, Any], ...],
        generation: int,
    ) -> None:
        self._owner = owner
        self._items = items
        self._generation = generation


class _PreparedMapping:
    __slots__ = ("_owner", "_snapshot", "_items", "_generation")

    def __init__(
        self,
        owner: "MappingRegistry",
        snapshot: _MappingSnapshot,
        items: dict[str, Any],
        generation: int,
    ) -> None:
        self._owner = owner
        self._snapshot = snapshot
        self._items = items
        self._generation = generation


class MappingRegistry:
    """Thread-safe string-keyed registry with opaque transaction views."""

    def __init__(
        self,
        surface: str,
        items: Optional[dict[str, Any]] = None,
        lock: Optional[threading.RLock] = None,
    ) -> None:
        self.surface = surface
        self._items = items if items is not None else {}
        self._lock = lock if lock is not None else threading.RLock()
        self._generation = 0
        self._frozen = False
        self._transaction_owner: Optional[MappingRegistry] = None
        self._transaction_snapshot: Optional[_MappingSnapshot] = None

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def get(self, key: str) -> Any:
        with self._lock:
            return self._items.get(key)

    def values(self) -> list[Any]:
        with self._lock:
            return list(self._items.values())

    def items_snapshot(self) -> tuple[tuple[str, Any], ...]:
        with self._lock:
            return tuple(self._items.items())

    def put(
        self,
        key: str,
        value: Any,
        *,
        replace: bool = True,
    ) -> tuple[bool, Any]:
        """Atomically add *value* after callers finish validation callbacks."""
        with self._lock:
            if self._frozen:
                raise RuntimeError(f"{self.surface} transaction view is frozen")
            existing = self._items.get(key)
            if existing is not None and not replace:
                return False, existing
            self._items[key] = value
            self._generation += 1
            return True, existing

    def remove(self, key: str) -> bool:
        with self._lock:
            if self._frozen:
                raise RuntimeError(f"{self.surface} transaction view is frozen")
            if key not in self._items:
                return False
            del self._items[key]
            self._generation += 1
            return True

    def clear(self) -> None:
        with self._lock:
            if self._frozen:
                raise RuntimeError(f"{self.surface} transaction view is frozen")
            self._items.clear()
            self._generation += 1

    def take_snapshot(self) -> _MappingSnapshot:
        with self._lock:
            return _MappingSnapshot(
                self,
                tuple(self._items.items()),
                self._generation,
            )

    def snapshot_items(
        self,
        snapshot: _MappingSnapshot,
    ) -> tuple[tuple[str, Any], ...]:
        self._validate_snapshot(snapshot)
        return snapshot._items

    def put_if_snapshot(
        self,
        snapshot: _MappingSnapshot,
        key: str,
        value: Any,
        *,
        replace: bool = True,
    ) -> tuple[bool, Any]:
        """Put only if the exact callback-free validation baseline is live."""
        self._validate_snapshot(snapshot)
        with self._lock:
            if self._frozen:
                raise RuntimeError(f"{self.surface} transaction view is frozen")
            if not self.matches_locked(snapshot):
                raise RegistryTransactionConflict(
                    self.surface,
                    snapshot._generation,
                    self._generation,
                )
            existing = self._items.get(key)
            if existing is not None and not replace:
                return False, existing
            self._items[key] = value
            self._generation += 1
            return True, existing

    def _validate_snapshot(self, snapshot: _MappingSnapshot) -> None:
        if not isinstance(snapshot, _MappingSnapshot) or snapshot._owner is not self:
            raise ValueError(
                f"transaction snapshot belongs to a different {self.surface} registry"
            )

    def create_view(
        self,
        snapshot: _MappingSnapshot,
        *,
        remove_keys: Iterable[str] = (),
    ) -> "MappingRegistry":
        self._validate_snapshot(snapshot)
        removed = frozenset(remove_keys)
        staged = MappingRegistry(
            self.surface,
            {
                key: value
                for key, value in snapshot._items
                if key not in removed
            },
        )
        staged._generation = snapshot._generation
        staged._transaction_owner = self
        staged._transaction_snapshot = snapshot
        return staged

    def prepare(
        self,
        snapshot: _MappingSnapshot,
        staged: "MappingRegistry",
    ) -> _PreparedMapping:
        self._validate_snapshot(snapshot)
        if (
            not isinstance(staged, MappingRegistry)
            or staged._transaction_owner is not self
            or staged._transaction_snapshot is not snapshot
        ):
            raise ValueError(
                f"transaction view belongs to a different {self.surface} registry"
            )
        with staged._lock:
            prepared = _PreparedMapping(
                self,
                snapshot,
                dict(staged._items),
                staged._generation,
            )
            staged._frozen = True
            return prepared

    def matches_locked(self, snapshot: _MappingSnapshot) -> bool:
        return (
            self._generation == snapshot._generation
            and len(self._items) == len(snapshot._items)
            and all(
                key in self._items and self._items[key] is value
                for key, value in snapshot._items
            )
        )

    def validate_prepared_locked(
        self,
        snapshot: _MappingSnapshot,
        prepared: _PreparedMapping,
    ) -> None:
        self._validate_snapshot(snapshot)
        if (
            not isinstance(prepared, _PreparedMapping)
            or prepared._owner is not self
            or prepared._snapshot is not snapshot
        ):
            raise ValueError(
                f"prepared state belongs to a different {self.surface} registry"
            )
        if not self.matches_locked(snapshot):
            raise RegistryTransactionConflict(
                self.surface,
                snapshot._generation,
                self._generation,
            )

    def install_prepared_locked(
        self,
        snapshot: _MappingSnapshot,
        prepared: _PreparedMapping,
    ) -> None:
        self.validate_prepared_locked(snapshot, prepared)
        self._items = dict(prepared._items)
        self._generation = max(self._generation, prepared._generation) + 1

    def restore_snapshot_locked(self, snapshot: _MappingSnapshot) -> None:
        """Restore an opaque snapshot exactly while the registry lock is held."""
        self._validate_snapshot(snapshot)
        self._items = dict(snapshot._items)
        self._generation = snapshot._generation


class RegistryTransactionSurface:
    """Opaque plugin transaction API owned by one module registry."""

    def __init__(
        self,
        state: MappingRegistry,
        register_fn: Callable[[MappingRegistry, Any], Optional[str]],
    ) -> None:
        self.surface = state.surface
        self._state = state
        self._register_fn = register_fn

    @property
    def lock(self) -> threading.RLock:
        return self._state.lock

    def take_snapshot(self) -> _MappingSnapshot:
        return self._state.take_snapshot()

    def create_view(
        self,
        snapshot: _MappingSnapshot,
        *,
        remove_keys: Iterable[str] = (),
    ) -> MappingRegistry:
        return self._state.create_view(snapshot, remove_keys=remove_keys)

    def register(self, staged: MappingRegistry, value: Any) -> Optional[str]:
        return self._register_fn(staged, value)

    def prepare(
        self,
        snapshot: _MappingSnapshot,
        staged: MappingRegistry,
    ) -> _PreparedMapping:
        return self._state.prepare(snapshot, staged)

    def validate_prepared_locked(
        self,
        snapshot: _MappingSnapshot,
        prepared: _PreparedMapping,
    ) -> None:
        self._state.validate_prepared_locked(snapshot, prepared)

    def install_prepared_locked(
        self,
        snapshot: _MappingSnapshot,
        prepared: _PreparedMapping,
    ) -> None:
        self._state.install_prepared_locked(snapshot, prepared)

    def restore_snapshot_locked(self, snapshot: _MappingSnapshot) -> None:
        self._state.restore_snapshot_locked(snapshot)
