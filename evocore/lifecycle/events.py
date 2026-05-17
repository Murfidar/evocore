"""Append-only optimizer lifecycle events."""

from evocore.results.generation import EventHistory, EventRecord, StopReason, append_run_stop_event

__all__ = ["EventHistory", "EventRecord", "StopReason", "append_run_stop_event"]
