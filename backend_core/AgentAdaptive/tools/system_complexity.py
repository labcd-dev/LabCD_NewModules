MIN_GRADE = 1
MAX_GRADE = 5

# states plus any inputs/outputs beyond 1-in/1-out (MIMO costs more), bucketed
# into grades 1-5 via _THRESHOLDS below
_THRESHOLDS = ((2, 1), (3, 2), (5, 3), (7, 4))


def complexity_raw(n_states, n_inputs, n_outputs):
    return (int(n_states) + max(0, int(n_inputs) - 1)
            + max(0, int(n_outputs) - 1))


def complexity_grade(n_states, n_inputs, n_outputs):
    raw = complexity_raw(n_states, n_inputs, n_outputs)
    for limit, grade in _THRESHOLDS:
        if raw <= limit:
            return grade
    return MAX_GRADE


def complexity_grade_from_spec(spec):
    dyn = (spec or {}).get("dynamics") or {}
    states = dyn.get("states") or []
    if not states:
        return None
    inputs = dyn.get("inputs") or []
    outputs = dyn.get("outputs") or []
    return complexity_grade(len(states), len(inputs), len(outputs))
