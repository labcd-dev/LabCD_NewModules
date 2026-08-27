import time


def _emit(on_event, **fields):
    if on_event is not None:
        fields.setdefault("ts", time.time())
        on_event(fields)


def _remap_note_stage(on_event, from_stage, to_stage):
    # _run_smc/_run_backstepping always hardcode stage="design": this relabels
    # it to "build" during a replay so it doesn't collide with the closed row
    if on_event is None:
        return on_event

    def wrapped(ev):
        if ev.get("kind") == "note" and ev.get("stage") == from_stage:
            ev = dict(ev, stage=to_stage)
        on_event(ev)
    return wrapped
