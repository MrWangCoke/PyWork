from __future__ import annotations

import pytest

from pywork.runtime.engine import RuntimeEngine
from pywork.runtime.events import RuntimeEventType
from pywork.tasks.task import TaskType


@pytest.mark.asyncio
async def test_task_manager_events_are_bridged_to_runtime_event_bus(tmp_path) -> None:
    engine = RuntimeEngine(
        config={
            "workspace": {
                "path": str(tmp_path),
                "project_root": str(tmp_path),
            }
        }
    )

    received = []

    def collect(event):
        received.append(event)

    engine.event_bus.subscribe(collect)

    record = await engine.task_manager.create_task(
        "Example task",
        task_type=TaskType.GENERIC,
    )

    task_events = [
        event
        for event in received
        if event.event_type == RuntimeEventType.STATUS
        and event.status == "task_created"
    ]

    assert task_events
    assert task_events[-1].metadata["task_event"] is True
    assert task_events[-1].metadata["task_id"] == record.id
    assert task_events[-1].metadata["task_event_type"] == "created"