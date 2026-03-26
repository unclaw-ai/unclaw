"""Helpers for compact mission execution ordering."""

from __future__ import annotations

from collections.abc import Sequence

from unclaw.core.mission_state import MissionDeliverableState, MissionState


def ordered_execution_queue(
    *,
    deliverables: Sequence[MissionDeliverableState],
    preferred_queue: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return a compact ordered queue of pending or active deliverables."""

    status_by_id = {
        deliverable.deliverable_id: deliverable.status for deliverable in deliverables
    }
    queue: list[str] = []
    for deliverable_id in preferred_queue:
        if status_by_id.get(deliverable_id) in {"pending", "active"}:
            queue.append(deliverable_id)
    for deliverable in deliverables:
        if (
            deliverable.status in {"pending", "active"}
            and deliverable.deliverable_id not in queue
        ):
            queue.append(deliverable.deliverable_id)
    return tuple(queue)


def resolve_active_deliverable_id(
    *,
    mission_state: MissionState,
    preferred_active_deliverable_id: str | None = None,
) -> str | None:
    """Return the active deliverable id from queue order and current status."""

    if preferred_active_deliverable_id is not None:
        deliverable = mission_state.get_deliverable(preferred_active_deliverable_id)
        if deliverable is not None and deliverable.status in {"pending", "active"}:
            return preferred_active_deliverable_id
    if mission_state.active_deliverable_id is not None:
        deliverable = mission_state.get_deliverable(mission_state.active_deliverable_id)
        if deliverable is not None and deliverable.status in {"pending", "active"}:
            return mission_state.active_deliverable_id
    queue = ordered_execution_queue(
        deliverables=mission_state.deliverables,
        preferred_queue=mission_state.execution_queue,
    )
    return queue[0] if queue else None


def missing_deliverable_ids(
    deliverables: Sequence[MissionDeliverableState],
) -> tuple[str, ...]:
    """Return the ids of every deliverable not yet verified completed."""

    return tuple(
        deliverable.deliverable_id
        for deliverable in deliverables
        if deliverable.status != "completed"
    )
