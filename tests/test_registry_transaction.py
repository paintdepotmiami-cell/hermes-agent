from registry_transaction import MappingRegistry


def test_install_prepared_locked_publishes_fresh_items_without_mutating_retained_reference():
    original = object()
    installed = object()
    registry = MappingRegistry("unit", {"old": original})
    retained_items = registry._items

    snapshot = registry.take_snapshot()
    staged = registry.create_view(snapshot)
    staged.put("new", installed)
    prepared = registry.prepare(snapshot, staged)

    with registry.lock:
        registry.install_prepared_locked(snapshot, prepared)

    assert retained_items == {"old": original}
    assert retained_items is not registry._items
    assert registry.items_snapshot() == (("old", original), ("new", installed))

    prepared._items["late"] = object()
    assert "late" not in dict(registry.items_snapshot())


def test_restore_snapshot_locked_restores_fresh_items_without_mutating_retained_reference():
    original = object()
    transient = object()
    registry = MappingRegistry("unit", {"old": original})
    snapshot = registry.take_snapshot()

    registry.put("transient", transient)
    retained_items = registry._items

    with registry.lock:
        registry.restore_snapshot_locked(snapshot)

    assert retained_items == {"old": original, "transient": transient}
    assert retained_items is not registry._items
    assert registry.items_snapshot() == (("old", original),)
