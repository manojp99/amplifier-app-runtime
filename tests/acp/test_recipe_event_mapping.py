"""Tests for recipe event mapping to ACP updates."""

from __future__ import annotations

from acp.schema import AgentPlanUpdate  # type: ignore[import-untyped]

from amplifier_app_runtime.acp.event_mapper import AmplifierToAcpEventMapper


class TestRecipeSessionStart:
    """Test recipe:session:start event mapping."""

    def test_map_recipe_session_start_creates_plan(self):
        """Test recipe session start creates initial plan."""
        mapper = AmplifierToAcpEventMapper()

        # Create mock event object
        class MockEvent:
            type = "recipe:session:start"
            properties = {
                "session_id": "recipe_123",
                "recipe_name": "code-review",
                "steps": [
                    {"name": "analysis", "status": "pending", "agent": "zen-architect"},
                    {"name": "feedback", "status": "pending", "agent": "self"},
                ],
            }

        result = mapper.map_event(MockEvent())

        assert result.update is not None
        assert isinstance(result.update, AgentPlanUpdate)
        assert len(result.update.entries) == 2
        assert result.update.entries[0].content == "1. analysis (zen-architect)"
        assert result.update.entries[0].status == "pending"
        assert result.update.entries[1].content == "2. feedback (self)"

    def test_map_recipe_session_start_caches_plan_state(self):
        """Test recipe session start initializes internal plan state."""
        mapper = AmplifierToAcpEventMapper()

        # Use flat dict format (no "properties" wrapper)
        event_data = {
            "type": "recipe:session:start",
            "session_id": "recipe_123",
            "steps": [
                {"name": "step1", "status": "pending", "agent": "self"},
            ],
        }

        mapper.map_event(event_data)

        # Verify internal state initialized
        assert len(mapper._current_plan) == 1
        assert mapper._recipe_session_id == "recipe_123"

    def test_map_recipe_session_start_empty_steps(self):
        """Test recipe session start with no steps returns no update."""
        mapper = AmplifierToAcpEventMapper()

        event_data = {
            "type": "recipe:session:start",
            "session_id": "recipe_123",
            "steps": []
        }

        result = mapper.map_event(event_data)

        assert result.update is None


class TestRecipeStepStart:
    """Test recipe:step:start event mapping."""

    def test_map_recipe_step_start_updates_status(self):
        """Test step start updates specific step to in_progress."""
        mapper = AmplifierToAcpEventMapper()

        # Initialize with session start
        mapper.map_event(
            {
                "type": "recipe:session:start",
                "session_id": "recipe_123",
                "steps": [
                    {"name": "step1", "status": "pending", "agent": "self"},
                    {"name": "step2", "status": "pending", "agent": "self"},
                ],
            }
        )

        # Start first step
        result = mapper.map_event(
            {
                "type": "recipe:step:start",
                "session_id": "recipe_123",
                "step_index": 0,
                "step_name": "step1",
            }
        )

        assert result.update is not None
        assert isinstance(result.update, AgentPlanUpdate)
        assert result.update.entries[0].status == "in_progress"
        assert result.update.entries[1].status == "pending"

    def test_map_recipe_step_start_invalid_index(self):
        """Test step start with out-of-bounds index is handled gracefully."""
        mapper = AmplifierToAcpEventMapper()

        # Initialize with one step
        mapper.map_event(
            {
                "type": "recipe:session:start",
                "session_id": "recipe_123",
                "steps": [{"name": "step1", "status": "pending", "agent": "self"}],
            }
        )

        # Try to start non-existent step
        result = mapper.map_event(
            {
                "type": "recipe:step:start",
                "session_id": "recipe_123",
                "step_index": 5,
                "step_name": "invalid",
            }
        )

        # Should still return update (plan unchanged)
        assert result.update is not None


class TestRecipeStepComplete:
    """Test recipe:step:complete event mapping."""

    def test_map_recipe_step_complete_updates_status(self):
        """Test step complete updates specific step to completed."""
        mapper = AmplifierToAcpEventMapper()

        # Initialize and start first step
        mapper.map_event(
            {
                "type": "recipe:session:start",
                "session_id": "recipe_123",
                "steps": [
                    {"name": "step1", "status": "pending", "agent": "self"},
                    {"name": "step2", "status": "pending", "agent": "self"},
                ],
            }
        )

        mapper.map_event(
            {"type": "recipe:step:start", "step_index": 0, "step_name": "step1"}
        )

        # Complete first step
        result = mapper.map_event(
            {
                "type": "recipe:step:complete",
                "session_id": "recipe_123",
                "step_index": 0,
                "step_name": "step1",
                "result": "Analysis complete",
            }
        )

        assert result.update is not None
        assert result.update.entries[0].status == "completed"
        assert result.update.entries[1].status == "pending"


class TestRecipeApprovalPending:
    """Test recipe:approval:pending event mapping."""

    def test_map_recipe_approval_pending_returns_plan(self):
        """Test approval pending returns current plan state."""
        mapper = AmplifierToAcpEventMapper()

        # Initialize plan
        mapper.map_event(
            {
                "type": "recipe:session:start",
                "session_id": "recipe_123",
                "steps": [{"name": "step1", "status": "pending", "agent": "self"}],
            }
        )

        # Trigger approval gate
        result = mapper.map_event(
            {
                "type": "recipe:approval:pending",
                "session_id": "recipe_123",
                "stage_name": "review",
                "prompt": "Approve review stage?",
                "timeout_seconds": 3600,
            }
        )

        assert result.update is not None
        assert isinstance(result.update, AgentPlanUpdate)


class TestRecipeSessionComplete:
    """Test recipe:session:complete event mapping."""

    def test_map_recipe_session_complete_marks_all_completed(self):
        """Test recipe completion marks remaining steps as completed."""
        mapper = AmplifierToAcpEventMapper()

        # Initialize and start step
        mapper.map_event(
            {
                "type": "recipe:session:start",
                "session_id": "recipe_123",
                "steps": [
                    {"name": "step1", "status": "pending", "agent": "self"},
                    {"name": "step2", "status": "pending", "agent": "self"},
                ],
            }
        )

        mapper.map_event(
            {"type": "recipe:step:start", "step_index": 0, "step_name": "step1"}
        )

        # Complete recipe
        result = mapper.map_event(
            {
                "type": "recipe:session:complete",
                "session_id": "recipe_123",
                "status": "success",
                "total_steps": 2,
                "duration_seconds": 120.5,
            }
        )

        assert result.update is not None
        # All steps should be marked completed
        assert all(entry.status == "completed" for entry in result.update.entries)

    def test_map_recipe_session_complete_clears_state(self):
        """Test recipe completion clears internal plan state."""
        mapper = AmplifierToAcpEventMapper()

        # Initialize plan
        mapper.map_event(
            {
                "type": "recipe:session:start",
                "session_id": "recipe_123",
                "steps": [{"name": "step1", "status": "pending", "agent": "self"}],
            }
        )

        assert len(mapper._current_plan) == 1
        assert mapper._recipe_session_id == "recipe_123"

        # Complete recipe
        mapper.map_event(
            {
                "type": "recipe:session:complete",
                "session_id": "recipe_123",
                "status": "success",
            }
        )

        # State should be cleared
        assert len(mapper._current_plan) == 0
        assert mapper._recipe_session_id is None


class TestRecipeEventSequence:
    """Test complete recipe event sequences."""

    def test_full_recipe_flow_sequence(self):
        """Test complete recipe execution sequence."""
        mapper = AmplifierToAcpEventMapper()

        # 1. Session start
        result1 = mapper.map_event(
            {
                "type": "recipe:session:start",
                "session_id": "recipe_123",
                "recipe_name": "test-flow",
                "steps": [
                    {"name": "step1", "status": "pending", "agent": "self"},
                    {"name": "step2", "status": "pending", "agent": "self"},
                    {"name": "step3", "status": "pending", "agent": "self"},
                ],
            }
        )

        assert result1.update is not None
        assert all(e.status == "pending" for e in result1.update.entries)

        # 2. Step 1 starts
        result2 = mapper.map_event(
            {"type": "recipe:step:start", "step_index": 0, "step_name": "step1"}
        )

        assert result2.update is not None
        assert result2.update.entries[0].status == "in_progress"
        assert result2.update.entries[1].status == "pending"
        assert result2.update.entries[2].status == "pending"

        # 3. Step 1 completes
        result3 = mapper.map_event(
            {
                "type": "recipe:step:complete",
                "step_index": 0,
                "step_name": "step1",
                "result": "Done",
            }
        )

        assert result3.update is not None
        assert result3.update.entries[0].status == "completed"
        assert result3.update.entries[1].status == "pending"
        assert result3.update.entries[2].status == "pending"

        # 4. Step 2 starts
        result4 = mapper.map_event(
            {"type": "recipe:step:start", "step_index": 1, "step_name": "step2"}
        )

        assert result4.update is not None
        assert result4.update.entries[0].status == "completed"
        assert result4.update.entries[1].status == "in_progress"
        assert result4.update.entries[2].status == "pending"

        # 5. Step 2 completes
        result5 = mapper.map_event(
            {"type": "recipe:step:complete", "step_index": 1, "step_name": "step2"}
        )

        assert result5.update is not None
        assert result5.update.entries[0].status == "completed"
        assert result5.update.entries[1].status == "completed"
        assert result5.update.entries[2].status == "pending"

        # 6. Step 3 starts
        result6 = mapper.map_event(
            {"type": "recipe:step:start", "step_index": 2, "step_name": "step3"}
        )

        assert result6.update is not None
        assert result6.update.entries[2].status == "in_progress"

        # 7. Step 3 completes
        result7 = mapper.map_event(
            {"type": "recipe:step:complete", "step_index": 2, "step_name": "step3"}
        )

        assert result7.update is not None
        assert result7.update.entries[2].status == "completed"

        # 8. Recipe completes
        result8 = mapper.map_event(
            {
                "type": "recipe:session:complete",
                "session_id": "recipe_123",
                "status": "success",
                "total_steps": 3,
                "duration_seconds": 45.2,
            }
        )

        # All completed
        assert result8.update is not None
        assert all(e.status == "completed" for e in result8.update.entries)

        # State cleared
        assert len(mapper._current_plan) == 0

    def test_recipe_with_approval_gate_sequence(self):
        """Test recipe with approval gate in sequence."""
        mapper = AmplifierToAcpEventMapper()

        # Start recipe
        mapper.map_event(
            {
                "type": "recipe:session:start",
                "session_id": "recipe_123",
                "steps": [
                    {"name": "planning", "status": "pending", "agent": "zen-architect"},
                    {"name": "implementation", "status": "pending", "agent": "modular-builder"},
                ],
            }
        )

        # Complete planning stage
        mapper.map_event(
            {"type": "recipe:step:start", "step_index": 0, "step_name": "planning"}
        )

        mapper.map_event(
            {
                "type": "recipe:step:complete",
                "step_index": 0,
                "step_name": "planning",
            }
        )

        # Approval gate before implementation
        result = mapper.map_event(
            {
                "type": "recipe:approval:pending",
                "session_id": "recipe_123",
                "stage_name": "implementation",
                "prompt": "Review planning output and approve implementation?",
            }
        )

        assert result.update is not None
        # First step completed, second still pending
        assert result.update.entries[0].status == "completed"
        assert result.update.entries[1].status == "pending"


class TestRecipeEventEdgeCases:
    """Test edge cases and error handling."""

    def test_recipe_step_before_session_start(self):
        """Test step event before session start is handled gracefully."""
        mapper = AmplifierToAcpEventMapper()

        # Try to start step without initializing plan
        result = mapper.map_event(
            {"type": "recipe:step:start", "step_index": 0, "step_name": "step1"}
        )

        # Should return update with empty plan
        assert result.update is not None
        assert len(result.update.entries) == 0

    def test_recipe_complete_without_start(self):
        """Test recipe complete without session start is handled."""
        mapper = AmplifierToAcpEventMapper()

        result = mapper.map_event(
            {
                "type": "recipe:session:complete",
                "session_id": "recipe_123",
                "status": "success",
            }
        )

        # Should return update (empty plan)
        assert result.update is not None

    def test_recipe_events_dont_break_todo_updates(self):
        """Test recipe events don't interfere with todo:update handling."""
        mapper = AmplifierToAcpEventMapper()

        # Start a recipe
        mapper.map_event(
            {
                "type": "recipe:session:start",
                "session_id": "recipe_123",
                "steps": [{"name": "step1", "status": "pending", "agent": "self"}],
            }
        )

        # Send todo:update (should work independently)
        result = mapper.map_event(
            {
                "type": "todo:update",
                "todos": [
                    {"content": "Task 1", "status": "in_progress"},
                    {"content": "Task 2", "status": "pending"},
                ]
            }
        )

        # Todo update should work
        assert result.update is not None
        assert isinstance(result.update, AgentPlanUpdate)
        assert len(result.update.entries) == 2
        assert result.update.entries[0].content == "Task 1"


class TestRecipeEventDict:
    """Test recipe events work with dict format (not just objects)."""

    def test_map_recipe_event_as_dict(self):
        """Test recipe events can be passed as plain dicts."""
        mapper = AmplifierToAcpEventMapper()

        # Pass event as dict instead of object
        event_dict = {
            "type": "recipe:session:start",
            "session_id": "recipe_123",
            "steps": [{"name": "step1", "status": "pending", "agent": "self"}],
        }

        result = mapper.map_event(event_dict)

        assert result.update is not None
        assert isinstance(result.update, AgentPlanUpdate)


class TestRecipeEventLogging:
    """Test event mapping logs appropriately."""

    def test_recipe_session_start_logs_info(self, caplog):
        """Test recipe session start logs info message."""
        import logging

        caplog.set_level(logging.DEBUG)

        mapper = AmplifierToAcpEventMapper()

        mapper.map_event(
            {
                "type": "recipe:session:start",
                "session_id": "recipe_123",
                "recipe_name": "test-recipe",
                "steps": [{"name": "step1", "status": "pending", "agent": "self"}],
            }
        )

        # Should log recipe start
        assert any("Recipe session started" in record.message for record in caplog.records)

    def test_recipe_complete_logs_duration(self, caplog):
        """Test recipe complete logs duration and status."""
        import logging

        caplog.set_level(logging.INFO)

        mapper = AmplifierToAcpEventMapper()

        # Initialize plan
        mapper.map_event(
            {
                "type": "recipe:session:start",
                "session_id": "recipe_123",
                "steps": [{"name": "step1", "status": "pending", "agent": "self"}],
            }
        )

        mapper.map_event(
            {
                "type": "recipe:session:complete",
                "session_id": "recipe_123",
                "status": "success",
                "total_steps": 3,
                "duration_seconds": 125.7,
            }
        )

        # Should log completion with stats
        assert any("Recipe session completed" in record.message for record in caplog.records)
        assert any("125.7" in record.message for record in caplog.records)
