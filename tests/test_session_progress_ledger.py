from __future__ import annotations

import pytest

from unclaw.core.mission_state import MissionState, MissionTaskState
from unclaw.core.session_manager import SessionManager
from unclaw.settings import load_settings

pytestmark = pytest.mark.integration


def test_progress_ledger_projects_active_mission_step(make_temp_project) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)

    try:
        session = session_manager.ensure_current_session()
        session_manager.persist_mission_state(
            MissionState(
                mission_id="mission-active",
                mission_goal="Research then write a note.",
                status="active",
                tasks=(
                    MissionTaskState(
                        id="t1",
                        title="Research the topic",
                        kind="web_research",
                        status="completed",
                        required_evidence=("full_web_research",),
                        satisfied_evidence=("full_web_research",),
                    ),
                    MissionTaskState(
                        id="t2",
                        title="Write the note",
                        kind="file_write",
                        status="active",
                        depends_on=("t1",),
                        required_evidence=("artifact_write", "artifact_readback"),
                    ),
                ),
                active_task_id="t2",
                updated_at="2026-03-27T09:00:00Z",
                next_expected_evidence="artifact_write, artifact_readback",
            ),
            session_id=session.id,
        )
        ledger = session_manager.get_session_progress_ledger(session.id)

        assert ledger
        assert ledger[-1].status == "active"
        assert ledger[-1].step == "Write the note"
        assert ledger[-1].detail == "mission step in progress"
    finally:
        session_manager.close()


def test_progress_ledger_projects_blocked_mission_blocker(make_temp_project) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)

    try:
        session = session_manager.ensure_current_session()
        session_manager.persist_mission_state(
            MissionState(
                mission_id="mission-blocked",
                mission_goal="Write the note.",
                status="blocked",
                tasks=(
                    MissionTaskState(
                        id="t1",
                        title="Write the note",
                        kind="file_write",
                        status="blocked",
                        required_evidence=("artifact_write", "artifact_readback"),
                        latest_error="permission denied",
                    ),
                ),
                active_task_id=None,
                updated_at="2026-03-27T09:00:00Z",
                blocker="permission denied",
                last_blocker="permission denied",
                executor_state="blocked",
            ),
            session_id=session.id,
        )
        ledger = session_manager.get_session_progress_ledger(session.id)

        assert ledger
        assert ledger[-1].status == "blocked"
        assert ledger[-1].step == "Write the note"
        assert ledger[-1].detail == "permission denied"
    finally:
        session_manager.close()
