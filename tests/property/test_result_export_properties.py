import json

from hypothesis import given
from hypothesis import strategies as st

from evocore.stats import EventHistory, EventRecord

json_scalars = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-1000, max_value=1000)
    | st.floats(
        min_value=-1000.0,
        max_value=1000.0,
        allow_nan=False,
        allow_infinity=False,
    )
    | st.text(max_size=20)
)
json_values = st.recursive(
    json_scalars,
    lambda children: (
        st.lists(children, max_size=3)
        | st.dictionaries(st.text(min_size=1, max_size=12), children, max_size=3)
    ),
    max_leaves=8,
)


@given(
    st.lists(
        st.dictionaries(st.text(min_size=1, max_size=12), json_values, max_size=3),
        max_size=8,
    )
)
def test_event_history_rows_are_json_round_trippable(metadata_rows):
    history = EventHistory()
    for index, metadata in enumerate(metadata_rows):
        history.append(
            EventRecord(
                event_index=index,
                event_type="tell",
                batch_id=f"b-{index}",
                candidate_id=f"c-{index}",
                candidate_hash=f"hash-{index}",
                confidence="trusted_full",
                raw_score=float(index),
                comparison_score=float(index),
                cost=1.0,
                status="trusted",
                origin="random",
                genes=(float(index), index),
                params={"x": float(index), "period": index},
                metadata=metadata,
            )
        )

    rows = history.to_rows()
    assert json.loads(json.dumps(rows, sort_keys=True, allow_nan=False)) == rows
