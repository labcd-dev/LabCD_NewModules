"""
================================================================================
agents/animation_agent.py
================================================================================
Animation Agent: replaces the earlier "static SVG schematic" feature. The
LLM's job here is narrow and specific, per design: given the system's
states/params, write the body of ONE small Python function --

    def draw_frame(ax, state, params, state_names):
        ...draws the physical configuration for this one instant...

-- and everything else (the animation loop, iterating over the ACTUAL best-
result trajectory frame by frame, saving to a GIF) is ordinary, deterministic
Python code in render_animation_gif below, not LLM output. This is a
meaningfully different (and much safer) shape than "ask the LLM to write a
whole animation script and run it": the LLM only ever produces one
constrained function body, which is validated (see _validate_and_sandbox)
before it's ever executed, and it only runs inside a restricted namespace
with no filesystem/network/process access.

SECURITY NOTE: this is the one place in the whole codebase that executes
LLM-generated Python. Treat any change here with real caution -- the
validation in _validate_and_sandbox is not optional defense-in-depth, it is
the only thing standing between "the model does something weird" and "the
model does something weird with access to the filesystem." Three
independent layers, all required to pass:
  1. AST structural check: the code must be EXACTLY one function definition
     named draw_frame with no other top-level statements.
  2. AST content check: walks the whole function body and rejects import
     statements, dunder attribute access (the standard sandbox-escape
     vector in Python -- e.g. reaching a class's __bases__ to climb back to
     builtins), and calls to a deny-list of dangerous names.
  3. Restricted execution: even validated code only ever runs against a
     hand-picked dict of safe builtins (no open/eval/exec/__import__/etc)
     and a small set of allowed modules (numpy, matplotlib primitives) --
     nothing else is reachable from inside the function no matter what the
     code says.
"""

from __future__ import annotations

import ast
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from ..utils.logging_utils import get_logger
from .llm_base import get_llm, invoke_with_retry

log = get_logger(__name__)

FUNCTION_NAME = "draw_frame"
ALLOWED_PARAMS = ("ax", "state", "params", "state_names")

# Anything reachable from these names can eventually reach the filesystem,
# network, process control, or the interpreter's own internals -- rejected
# outright if the generated code references them anywhere, as a name OR as
# an attribute.
FORBIDDEN_NAMES = {
    "os", "sys", "subprocess", "socket", "shutil", "pickle", "importlib",
    "open", "exec", "eval", "compile", "__import__", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "input", "breakpoint",
    "memoryview", "help", "exit", "quit", "reload",
}


class AnimationSpec(BaseModel):
    draw_frame_code: str = Field(
        description=f"Python source for EXACTLY one function definition, nothing else before or after it: "
        f"'def {FUNCTION_NAME}(ax, state, params, state_names):' followed by its body. Draw the physical "
        f"configuration of the system at this one instant onto `ax` (a matplotlib Axes, already created for "
        f"you -- do not create a new figure/axes). `state` is a 1D numpy array with the CURRENT value of "
        f"every state variable, in the same order as `state_names` (a list of strings). `params` is a dict "
        f"of the system's physical parameters (mass, length, etc, by name). Call ax.clear() first, then use "
        f"ONLY matplotlib Axes drawing methods available on `ax` (ax.plot, ax.scatter, ax.add_patch with "
        f"shapes from the `patches` module already provided to you -- patches.Circle, patches.Rectangle, "
        f"patches.FancyArrow -- ax.text, ax.set_xlim, ax.set_ylim, ax.set_aspect, ax.set_title, ax.axis). "
        f"Use the SAME xlim/ylim every call (pick fixed values that comfortably fit the system's full range "
        f"of motion) so the animation doesn't jitter/rescale between frames. Do not use any import "
        f"statement -- numpy is already available as `np`, matplotlib.patches as `patches`. Do not read or "
        f"write files, use the network, or call exec/eval/compile/open/__import__ -- none of that is "
        f"available anyway, but do not attempt it.",
    )
    is_3d: bool = Field(description="Whether `ax` should be a 3D axes (mpl_toolkits.mplot3d) -- True for "
                         "genuinely spatial systems like multirotor aircraft, False for planar systems like "
                         "pendulums, carts, or anything that only moves in a single vertical plane.")
    description: str = Field(description="1-2 sentence caption describing what the animation shows.")


ANIMATION_PROMPT_TEMPLATE = """
You are the Animation Agent. Write the body of a single Python function that
draws ONE FRAME of an animation of the physical system described below, at a
given instant in time (given its current state vector).

System class name: "{class_name}"
States ({n_states}): {state_names}
Inputs ({n_inputs}): {input_names}
Physical parameters: {params}
{user_context_block}
Reason briefly about what kind of physical system this most likely is (the
same way you would for a textbook figure -- e.g. "cart_pos"/"pole_angle"
strongly suggests a cart-pole, "theta1"/"theta2" a double pendulum, multiple
rotor-style inputs with x/y/z/phi/theta/psi states a multirotor aircraft)
before writing the drawing code, so the picture is geometrically sensible --
e.g. an actual pendulum ROD of the correct length attached at the correct
pivot, not just an abstract dot.

IMPORTANT and easy to get wrong without more information: which physical
configuration a zero angle corresponds to is a MODELING CONVENTION, not
something you can always safely assume -- e.g. for an "inverted pendulum,"
theta=0 conventionally means the UPRIGHT (unstable) position, not hanging
down, which is the opposite of a plain pendulum's usual convention. If the
context above doesn't clarify this, prefer a NEUTRAL default that's
unlikely to actively mislead (angles measured from the vertical, drawn
literally at face value: theta radians from straight up) rather than
guessing "hanging down" for anything with "pendulum" in the name.

Keep the drawing itself simple (a small number of shapes: lines for rods,
circles for masses/joints, a rectangle for a cart) -- this is a clear,
readable schematic in motion, not a detailed illustration.
""".strip()

animation_prompt = PromptTemplate(
    input_variables=["class_name", "n_states", "n_inputs", "state_names", "input_names", "params", "user_context_block"],
    template=ANIMATION_PROMPT_TEMPLATE,
)


class AnimationCodeRejected(Exception):
    """Raised by _validate_and_sandbox when generated code fails any safety check."""


def _validate_and_sandbox(code: str) -> Any:
    """The three-layer check described in the module docstring. Returns the
    validated, callable draw_frame function on success. Raises
    AnimationCodeRejected (with a human-readable reason) on any failure --
    callers should treat this as "no animation," not attempt to salvage or
    partially execute anything.
    """
    code = code.strip()
    # LLMs commonly wrap code in markdown fences even when asked not to.
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    code = code.strip()

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise AnimationCodeRejected(f"generated code has a syntax error: {e}")

    # Layer 1: structural -- exactly one statement, a FunctionDef named
    # draw_frame with the expected parameters, nothing else at module level.
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise AnimationCodeRejected("expected exactly one function definition and nothing else")
    func = tree.body[0]
    if func.name != FUNCTION_NAME:
        raise AnimationCodeRejected(f"function must be named {FUNCTION_NAME!r}, got {func.name!r}")
    param_names = [a.arg for a in func.args.args]
    if param_names != list(ALLOWED_PARAMS):
        raise AnimationCodeRejected(f"function signature must be exactly {ALLOWED_PARAMS}, got {tuple(param_names)}")

    # Layer 2: content -- walk the entire body, reject anything that could
    # reach outside the sandbox.
    for node in ast.walk(func):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise AnimationCodeRejected("import statements are not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            raise AnimationCodeRejected(f"dunder attribute access is not allowed: .{node.attr}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise AnimationCodeRejected(f"use of forbidden name: {node.id}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_NAMES:
            raise AnimationCodeRejected(f"call to forbidden function: {node.func.id}")

    # Layer 3: restricted execution -- even validated code only ever sees a
    # hand-picked, minimal namespace. No filesystem, no network, no process
    # control, no interpreter internals are reachable from inside it no
    # matter what the code contains, regardless of the checks above.
    import numpy as np
    import matplotlib.patches as patches

    SAFE_BUILTINS = {
        "range": range, "len": len, "min": min, "max": max, "abs": abs, "round": round,
        "int": int, "float": float, "str": str, "bool": bool, "list": list, "tuple": tuple,
        "dict": dict, "set": set, "enumerate": enumerate, "zip": zip, "sum": sum, "sorted": sorted,
        "True": True, "False": False, "None": None, "print": lambda *a, **k: None,  # no-op, not silently erroring
    }
    exec_globals: Dict[str, Any] = {"__builtins__": SAFE_BUILTINS, "np": np, "patches": patches}
    try:
        exec(compile(tree, "<animation_agent>", "exec"), exec_globals)  # noqa: S102 -- see module docstring
    except Exception as e:  # noqa: BLE001
        raise AnimationCodeRejected(f"generated code failed to define the function: {e}")

    fn = exec_globals.get(FUNCTION_NAME)
    if fn is None or not callable(fn):
        raise AnimationCodeRejected("draw_frame was not defined after executing the generated code")
    return fn


def generate_animation_code(
    class_name: str,
    state_names: List[str],
    input_names: List[str],
    params: Dict[str, Any],
    user_context: str = "",
    tracker: Optional[Any] = None,
) -> Tuple[Optional[Any], bool, str]:
    """Returns (draw_frame_fn_or_None, is_3d, note). draw_frame_fn is None
    if generation OR validation failed for any reason -- callers should
    show `note` (which explains why) and not attempt to render anything.

    ``user_context``: optional free-text the user can supply (e.g. "theta=0
    is the UPRIGHT/unstable equilibrium, not hanging down" or "positive
    torque is counterclockwise") to correct for the kind of thing the model
    genuinely cannot reliably infer on its own from state/input names alone
    -- sign conventions and which configuration is the modeled equilibrium
    are modeling choices, not physical facts derivable from names.
    """
    try:
        llm = get_llm().with_structured_output(AnimationSpec)
        user_context_block = (
            f"\nUser-provided context (trust this over your own guesses -- the person who built this "
            f"system knows its conventions better than you can infer from names alone):\n{user_context.strip()}\n"
            if user_context and user_context.strip() else ""
        )
        prompt_text = animation_prompt.format(
            class_name=class_name, n_states=len(state_names), n_inputs=len(input_names),
            state_names=", ".join(state_names) or "unknown",
            input_names=", ".join(input_names) or "unknown",
            params=params or "none declared",
            user_context_block=user_context_block,
        )
        result: AnimationSpec = invoke_with_retry(llm, prompt_text, max_retries=1, node_name="AnimationAgent",
                                                     tracker=tracker)
    except Exception as e:  # noqa: BLE001
        log.error("[AnimationAgent] LLM call failed: %s", e)
        return None, False, f"Animation generation failed: {e}"

    try:
        fn = _validate_and_sandbox(result.draw_frame_code)
    except AnimationCodeRejected as e:
        log.warning("[AnimationAgent] generated code rejected: %s", e)
        return None, False, f"The generated animation code didn't pass safety validation ({e}), so it wasn't run."

    return fn, result.is_3d, result.description


def render_animation_gif(
    draw_frame_fn: Any,
    is_3d: bool,
    states: Any,  # np.ndarray, shape (n_steps, n_states)
    times: Any,   # np.ndarray, shape (n_steps,)
    params: Dict[str, Any],
    state_names: List[str],
    max_frames: int = 90,
    max_seconds: float = 90.0,
) -> Tuple[Optional[bytes], str]:
    """Runs the validated draw_frame function once per (subsampled) frame
    of the ACTUAL best-result trajectory and saves the result as an
    in-memory GIF. Returns (gif_bytes_or_None, note).

    The AST validation in _validate_and_sandbox rules out sandbox escapes,
    but can't rule out a pathological infinite loop WITHIN a single frame's
    drawing code (not a security issue -- the sandbox still holds -- but an
    availability one: a single stuck call would otherwise hang this whole
    function, and with it the Streamlit app, forever). Running the actual
    rendering in a worker thread with a hard wait-timeout means this
    function always returns within max_seconds one way or another, even in
    that worst case -- the worker thread itself may still be stuck running
    in the background afterward, but the app regains control and can
    report the failure instead of freezing.
    """
    import concurrent.futures

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(_render_animation_gif_inner, draw_frame_fn, is_3d, states, times,
                          params, state_names, max_frames, max_seconds)
    try:
        result = future.result(timeout=max_seconds + 10.0)  # grace period beyond the inner budget
        pool.shutdown(wait=False)
        return result
    except concurrent.futures.TimeoutError:
        log.error("[AnimationAgent] rendering exceeded the hard timeout -- a frame likely hung.")
        # wait=False is deliberate: the worker thread is genuinely stuck (an
        # infinite loop cannot be interrupted from outside in Python), and
        # the context-manager-style `with ThreadPoolExecutor(...)` pattern
        # tried here first calls shutdown(wait=True) on exit, which BLOCKS
        # the caller (and with it, the whole app) until that stuck thread
        # finishes -- which it never will. wait=False lets this function
        # return promptly instead; the abandoned thread keeps running in
        # the background for the lifetime of the app process, which is a
        # real but much smaller cost than hanging the app entirely.
        pool.shutdown(wait=False)
        return None, (f"Animation rendering didn't finish within {max_seconds + 10:.0f}s -- the generated "
                       f"drawing code likely got stuck on a single frame (an infinite loop or similar). "
                       f"Try again; if it keeps happening, this system's animation may not be reliably "
                       f"generatable.")


def _render_animation_gif_inner(
    draw_frame_fn: Any,
    is_3d: bool,
    states: Any,
    times: Any,
    params: Dict[str, Any],
    state_names: List[str],
    max_frames: int,
    max_seconds: float,
) -> Tuple[Optional[bytes], str]:
    import os
    import tempfile

    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.figure import Figure  # NOT pyplot's plt.figure() -- this avoids matplotlib's
                                             # shared global pyplot state entirely, which is the
                                             # recommended pattern for rendering from a background
                                             # thread (this function runs inside a ThreadPoolExecutor
                                             # worker -- see render_animation_gif above) alongside a
                                             # main thread (Streamlit's own script execution) that may
                                             # ALSO be calling pyplot-based plotting concurrently.

    n_steps = len(states)
    if n_steps == 0:
        return None, "No trajectory data to animate."

    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d" if is_3d else None)
    fig.patch.set_facecolor("white")

    # ---- Probe: measure the REAL cost of a handful of frames (drawing +
    # this backend's per-frame overhead) before committing to a frame
    # count, rather than requesting the full max_frames and hoping it fits
    # the time budget. A fixed per-frame time budget derived from this
    # probe is a much more reliable way to stay within max_seconds than
    # trying to bail out mid-render (which -- the previous version of this
    # function tried exactly that from inside FuncAnimation's per-frame
    # callback, and it didn't actually save any time: skipping the DRAWING
    # after the budget was up still left the same number of frames for
    # PillowWriter to encode and write out). ----
    probe_start = time.time()
    n_probe = min(3, n_steps)
    probe_indices = [int(i * (n_steps - 1) / max(n_probe - 1, 1)) for i in range(n_probe)]
    for idx in probe_indices:
        draw_frame_fn(ax, states[idx], params, state_names)
    probe_elapsed = time.time() - probe_start
    per_frame_budget = (probe_elapsed / n_probe) if n_probe else 0.05
    # Encoding (PillowWriter) adds its own overhead on top of drawing --
    # budget generously for it (2x the measured draw-only cost) rather than
    # assuming drawing time alone predicts total time per frame.
    per_frame_total_estimate = max(per_frame_budget * 2.0, 0.01)
    # Reserve a fixed slice of the budget for the encode/write step at the
    # end, then size the frame count to fit comfortably in what's left.
    usable_seconds = max(max_seconds - probe_elapsed - 3.0, 1.0)
    safe_frame_count = max(2, min(max_frames, int(usable_seconds / per_frame_total_estimate)))

    step = max(1, n_steps // safe_frame_count)
    frame_indices = list(range(0, n_steps, step))
    if frame_indices[-1] != n_steps - 1:
        frame_indices.append(n_steps - 1)
    frame_indices = frame_indices[:safe_frame_count]  # hard cap regardless of rounding above

    rendered_indices: List[int] = []

    def update(i: int):
        idx = frame_indices[i]
        draw_frame_fn(ax, states[idx], params, state_names)
        ax.set_title(f"t = {times[idx]:.2f}s", fontsize=10)
        rendered_indices.append(idx)
        return []

    try:
        duration_s = float(times[frame_indices[-1]] - times[frame_indices[0]]) or 1.0
        fps = max(1, min(30, int(len(frame_indices) / duration_s)))
        anim = FuncAnimation(fig, update, frames=len(frame_indices), blit=False)
        # Animation.save() needs an actual file path (unlike Figure.savefig,
        # it doesn't accept a BytesIO buffer directly) -- write to a temp
        # file and read the bytes back.
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".gif")
        os.close(tmp_fd)
        try:
            anim.save(tmp_path, writer=PillowWriter(fps=fps))
            with open(tmp_path, "rb") as f:
                gif_bytes = f.read()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:  # noqa: BLE001
        log.error("[AnimationAgent] rendering failed: %s", e)
        return None, f"Rendering the animation failed: {e}"

    if safe_frame_count < min(max_frames, n_steps):
        note = (f"Rendered {len(rendered_indices)} frames covering {duration_s:.2f}s of simulated time "
                f"(fewer than the usual {max_frames}, based on how long each frame took to draw, to stay "
                f"within the time budget).")
    else:
        note = f"Rendered {len(rendered_indices)} frames covering {duration_s:.2f}s of simulated time."
    return gif_bytes, note
