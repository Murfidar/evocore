"""Private vNext batch ledger helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

from evocore import _core
from evocore.evaluation import EvaluationRecord
from evocore.exceptions import FitnessError


def batch_id_from_seed(master_seed: int, event_index: int) -> str:
    """Return a deterministic public batch ID for an ask event."""
    candidate_style_id = _core.candidate_id(int(master_seed), int(event_index), 0)
    return f"b-{candidate_style_id.removeprefix('c-')}"


@dataclass
class CandidateBatch:
    """Track records received for one ask() batch."""

    batch_id: str
    candidate_ids: tuple[str, ...]
    continuous_samples_by_id: dict[str, list[float]] = field(default_factory=dict)
    records_by_key: dict[tuple[str, str], EvaluationRecord] = field(default_factory=dict)
    consumed: bool = False
    _candidate_id_set: set[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._candidate_id_set = set(self.candidate_ids)

    def accept_record(
        self,
        record: EvaluationRecord,
        *,
        reject_consumed_trusted: bool = False,
    ) -> None:
        """Validate and store one record for this batch."""
        if reject_consumed_trusted and self.consumed and record.confidence == "trusted_full":
            raise FitnessError(f"batch {self.batch_id!r} has already been consumed.")
        if record.batch_id is not None and record.batch_id != self.batch_id:
            raise FitnessError(
                f"EvaluationRecord batch_id {record.batch_id!r} does not match "
                f"candidate batch {self.batch_id!r}."
            )
        if record.candidate_id not in self._candidate_id_set:
            raise FitnessError(
                f"candidate_id {record.candidate_id!r} does not belong to batch {self.batch_id!r}."
            )
        if record.confidence == "trusted_full":
            for existing in self.records_by_key.values():
                if (
                    existing.candidate_id == record.candidate_id
                    and existing.confidence == "trusted_full"
                ):
                    raise FitnessError(
                        f"candidate_id {record.candidate_id!r} already has a trusted_full record "
                        f"for batch {self.batch_id!r}."
                    )
        key = (record.candidate_id, record.rung)
        if key in self.records_by_key:
            raise FitnessError(
                f"candidate_id {record.candidate_id!r} already has a record for rung "
                f"{record.rung!r} in batch {self.batch_id!r}."
            )
        self.records_by_key[key] = record

    def ordered_trusted_full_records(self) -> list[EvaluationRecord] | None:
        """Return trusted records in ask order once the batch is complete."""
        trusted_by_candidate: dict[str, EvaluationRecord] = {}
        for record in self.records_by_key.values():
            if record.confidence == "trusted_full":
                trusted_by_candidate[record.candidate_id] = record
        if any(candidate_id not in trusted_by_candidate for candidate_id in self.candidate_ids):
            return None
        return [trusted_by_candidate[candidate_id] for candidate_id in self.candidate_ids]
