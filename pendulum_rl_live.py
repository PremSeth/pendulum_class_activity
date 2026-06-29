"""Live reinforcement-learning activity for the CartPole inverted pendulum.

Run the interactive app with:

    streamlit run pendulum_rl_live.py

Run a tiny non-interactive check with:

    python pendulum_rl_live.py --smoke-test
"""

from __future__ import annotations

import argparse
import base64
import io
import importlib
import importlib.util
import json
import math
import os
import pickle
import random
import sys
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


os.environ.setdefault("MPLCONFIGDIR", "/tmp/pendulum_rl_matplotlib")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


CART_HALF_WIDTH = 0.20
CART_HALF_HEIGHT = 0.12
CART_AXLE_Y = 0.06
POLE_LENGTH = 1.0
POLE_HALF_WIDTH = 0.04
ANIMAL_Y = 0.0


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def wrap_angle(theta: float) -> float:
    return ((float(theta) + math.pi) % (2.0 * math.pi)) - math.pi


def normalized_observation_values(
    raw_state: Any,
    animal_position: float = 0.0,
) -> dict[str, float]:
    """Map observations into stable ranges before the agent sees them."""
    x, x_dot, theta, theta_dot = [float(value) for value in raw_state]
    wrapped_theta = wrap_angle(theta)
    animal_distance = x - float(animal_position)

    return {
        "cart_position": clamp(x / 2.4),
        "cart_velocity": clamp(x_dot / 3.0),
        "pole_angle": clamp(wrapped_theta / math.pi),
        "sin_theta": math.sin(theta),
        "cos_theta": math.cos(theta),
        "pole_angular_velocity": clamp(theta_dot / 3.5),
        "animal_distance": clamp(animal_distance / 2.4),
        "abs_animal_distance": clamp(abs(animal_distance) / 2.4, 0.0, 1.0),
    }


# ---------------------------------------------------------------------------
# LIVE CODING SPOTS: change these functions during the activity.
# ---------------------------------------------------------------------------
def observation_function(
    raw_state: Any,
    features: tuple[str, ...],
    animal_position: float = 0.0,
) -> Any:
    """Choose what the agent sees from the CartPole state.

    Values returned here are normalized so Q-learning and DQN both train on
    compact inputs instead of raw environment magnitudes.
    """
    np = require_dependencies("numpy")["numpy"]
    feature_values = normalized_observation_values(raw_state, animal_position)
    return np.array([feature_values[name] for name in features], dtype=np.float32)


def action_function(action_index: int, action_forces: tuple[float, ...]) -> float:
    """Map the action chosen by the agent into a cart force."""
    return float(action_forces[action_index])


def reward_signal_observations(
    raw_state: Any,
    env_reward: float,
    terminated: bool,
    action_force: float,
    animal_position: float = 0.0,
    animal_radius: float = 0.0,
) -> dict[str, float]:
    """Base normalized signals used by reward blocks."""
    x, x_dot, theta, theta_dot = [float(value) for value in raw_state]
    wrapped_theta = wrap_angle(theta)
    animal_distance = x - animal_position
    contact = animal_contact(raw_state, animal_position, animal_radius)
    cart_near_width = max(0.01, animal_radius + CART_HALF_WIDTH)
    pole_near_width = max(0.01, animal_radius + POLE_HALF_WIDTH)
    track_limit = 2.4

    state_signals = {
        "cart_position": clamp(x / 2.4),
        "abs_cart_position": clamp(abs(x) / 2.4, 0.0, 1.0),
        "cart_velocity": clamp(x_dot / 3.0),
        "abs_cart_velocity": clamp(abs(x_dot) / 3.0, 0.0, 1.0),
        "pole_angle": clamp(wrapped_theta / math.pi),
        "abs_pole_angle": clamp(abs(wrapped_theta) / math.pi, 0.0, 1.0),
        "sin_theta": math.sin(theta),
        "cos_theta": math.cos(theta),
        "pole_angular_velocity": clamp(theta_dot / 3.5),
        "abs_pole_angular_velocity": clamp(abs(theta_dot) / 3.5, 0.0, 1.0),
        "action_force": clamp(action_force / 10.0),
        "abs_action_force": clamp(abs(action_force) / 10.0, 0.0, 1.0),
        "animal_distance": clamp(animal_distance / 2.4),
        "abs_animal_distance": clamp(abs(animal_distance) / 2.4, 0.0, 1.0),
        "pole_distance_to_animal": clamp(float(contact["pole_distance"]) / 2.4, 0.0, 1.0),
    }

    return {
        "env_reward": float(env_reward),
        "alive": 1.0,
        "cart_off_screen": 1.0 if abs(x) >= track_limit else 0.0,
        "fell": 1.0 if terminated else 0.0,
        "near_animal": max(0.0, 1.0 - float(contact["cart_distance"]) / cart_near_width),
        "near_pole_touch": max(0.0, 1.0 - float(contact["pole_distance"]) / pole_near_width),
        "cart_hit_animal": 1.0 if contact["cart_hit"] else 0.0,
        "pole_hit_animal": 1.0 if contact["pole_hit"] else 0.0,
        "hit_animal": 1.0 if contact["hit"] else 0.0,
        **state_signals,
    }


def animal_contact(
    raw_state: Any,
    animal_position: float,
    animal_radius: float,
) -> dict[str, float | bool]:
    """Detect full cart rectangle and pole capsule contact with the animal circle."""
    x, _, theta, _ = [float(value) for value in raw_state]
    animal_radius = max(0.0, float(animal_radius))
    animal_x = float(animal_position)
    animal_y = ANIMAL_Y

    cart_left = x - CART_HALF_WIDTH
    cart_right = x + CART_HALF_WIDTH
    cart_bottom = -CART_HALF_HEIGHT
    cart_top = CART_HALF_HEIGHT
    closest_cart_x = min(max(animal_x, cart_left), cart_right)
    closest_cart_y = min(max(animal_y, cart_bottom), cart_top)
    cart_surface_distance = math.hypot(animal_x - closest_cart_x, animal_y - closest_cart_y)
    cart_clearance = max(0.0, cart_surface_distance - animal_radius)
    cart_hit = animal_radius > 0.0 and cart_surface_distance <= animal_radius

    pole_base_x = x
    pole_base_y = CART_AXLE_Y
    pole_tip_x = pole_base_x + POLE_LENGTH * math.sin(theta)
    pole_tip_y = pole_base_y + POLE_LENGTH * math.cos(theta)
    segment_dx = pole_tip_x - pole_base_x
    segment_dy = pole_tip_y - pole_base_y
    segment_length_squared = max(1e-9, segment_dx * segment_dx + segment_dy * segment_dy)
    projection = ((animal_x - pole_base_x) * segment_dx + (animal_y - pole_base_y) * segment_dy) / segment_length_squared
    projection = max(0.0, min(1.0, projection))
    closest_pole_x = pole_base_x + projection * segment_dx
    closest_pole_y = pole_base_y + projection * segment_dy
    pole_centerline_distance = math.hypot(animal_x - closest_pole_x, animal_y - closest_pole_y)
    pole_clearance = max(0.0, pole_centerline_distance - POLE_HALF_WIDTH - animal_radius)
    pole_hit = animal_radius > 0.0 and pole_centerline_distance <= animal_radius + POLE_HALF_WIDTH

    return {
        "cart_hit": cart_hit,
        "pole_hit": pole_hit,
        "hit": cart_hit or pole_hit,
        "cart_distance": cart_clearance,
        "pole_distance": pole_clearance,
        "pole_centerline_distance": pole_centerline_distance,
    }


def reward_function(
    obs: Any,
    action: int,
    action_force: float,
    next_obs: Any,
    env_reward: float,
    terminated: bool,
    truncated: bool,
    weights: dict[str, Any],
) -> float:
    """Turn the environment reward into the reward the agent actually learns."""
    del obs, truncated, action

    signals = reward_signal_observations(
        next_obs,
        env_reward,
        terminated,
        action_force,
        float(weights.get("animal_position", 0.0)),
        float(weights.get("animal_radius", 0.0)),
    )
    x, _, theta, _ = [float(value) for value in next_obs]
    if "target_cart_position" in weights:
        target_x = float(weights.get("target_cart_position", 0.0))
        signals["target_cart_position_error"] = clamp(abs(x - target_x) / 2.4, 0.0, 1.0)
    if "target_pole_angle" in weights:
        target_theta = float(weights.get("target_pole_angle", 0.0))
        signals["target_pole_angle_error"] = clamp(abs(wrap_angle(theta - target_theta)) / math.pi, 0.0, 1.0)

    reward_tokens = weights.get("reward_tokens")
    if isinstance(reward_tokens, list) and reward_tokens:
        return evaluate_reward_tokens(reward_tokens, signals)

    total = 0.0
    product_group = 1.0
    group_started = False
    for term in weights["reward_terms"]:
        term_value = reward_term_value(term, signals)
        connector = str(term.get("connector", "add"))
        if connector == "multiply" and group_started:
            product_group *= term_value
        else:
            if group_started:
                total += product_group
            product_group = term_value
            group_started = True

    if group_started:
        total += product_group
    return float(total)


def default_reward_scale(signal: str) -> str:
    return "pi" if signal in {"pole_angle", "abs_pole_angle"} else "unit"


def clean_reward_scale(signal: str, scale: str) -> str:
    if signal not in REWARD_SCALE_SIGNALS:
        return "unit"
    if scale not in REWARD_SCALE_LABELS:
        return default_reward_scale(signal)
    return scale


def reward_signal_scale(value: float, scale: str) -> float:
    if scale == "pi":
        return float(value) * math.pi
    return float(value)


def reward_term_value(term: dict[str, Any], signals: dict[str, float]) -> float:
    signal = str(term["signal"])
    value = signals[signal]
    scale = clean_reward_scale(signal, str(term.get("scale", default_reward_scale(signal))))
    value = reward_signal_scale(value, scale)
    transform = str(term.get("transform", "abs" if term.get("absolute", False) else "raw"))
    if transform == "abs":
        value = abs(value)
    elif transform == "sin":
        value = math.sin(value)
    elif transform == "cos":
        value = math.cos(value)
    return float(term["factor"]) * value


def evaluate_reward_tokens(tokens: list[Any], signals: dict[str, float]) -> float:
    precedence = {"add": 1, "multiply": 2}
    output: list[float | str] = []
    operators: list[str] = []
    expecting_value = True

    def apply_operator_stack(new_operator: str) -> None:
        while (
            operators
            and operators[-1] != "("
            and not operators[-1].startswith("func:")
            and precedence[operators[-1]] >= precedence[new_operator]
        ):
            output.append(operators.pop())
        operators.append(new_operator)

    for token in tokens:
        if not isinstance(token, dict):
            continue

        token_type = str(token.get("type", "term" if "signal" in token else ""))
        if token_type == "term":
            try:
                output.append(reward_term_value(token, signals))
                expecting_value = False
            except (KeyError, TypeError, ValueError):
                continue
        elif token_type == "func" and isinstance(token.get("children"), list):
            child_value = evaluate_reward_tokens(token["children"], signals)
            function_name = str(token.get("func", "abs"))
            threshold = float(token.get("threshold", 0.0))
            if function_name == "abs":
                child_value = abs(child_value)
            elif function_name == "sin":
                child_value = math.sin(child_value)
            elif function_name == "cos":
                child_value = math.cos(child_value)
            elif function_name == "min":
                child_value = min(child_value, threshold)
            elif function_name == "max":
                child_value = max(child_value, threshold)
            else:
                continue
            output.append(float(token.get("factor", 1.0)) * child_value)
            expecting_value = False
        elif token_type == "func":
            function_name = str(token.get("func", "abs"))
            if function_name in ("abs", "sin", "cos"):
                operators.append(f"func:{function_name}")
                expecting_value = True
        elif token_type == "op" and not expecting_value:
            operator = "multiply" if token.get("op") in ("multiply", "*", "x") else "add"
            apply_operator_stack(operator)
            expecting_value = True
        elif token_type == "paren" and token.get("value") == "(":
            operators.append("(")
            expecting_value = True
        elif token_type == "paren" and token.get("value") == ")" and not expecting_value:
            while operators and operators[-1] != "(":
                output.append(operators.pop())
            if operators and operators[-1] == "(":
                operators.pop()
            if operators and operators[-1].startswith("func:"):
                output.append(operators.pop())
            expecting_value = False

    while operators:
        operator = operators.pop()
        if operator != "(":
            output.append(operator)

    stack: list[float] = []
    for item in output:
        if isinstance(item, float):
            stack.append(item)
        elif item in precedence and len(stack) >= 2:
            right = stack.pop()
            left = stack.pop()
            stack.append(left + right if item == "add" else left * right)
        elif isinstance(item, str) and item.startswith("func:") and stack:
            value = stack.pop()
            function_name = item.removeprefix("func:")
            if function_name == "abs":
                stack.append(abs(value))
            elif function_name == "sin":
                stack.append(math.sin(value))
            elif function_name == "cos":
                stack.append(math.cos(value))

    return float(stack[-1]) if stack else 0.0


REWARD_SIGNAL_LABELS: dict[str, str] = {
    "alive": "alive each step",
    "cart_position": "cart position",
    "abs_cart_position": "|cart position|",
    "cart_off_screen": "cart off screen",
    "cart_velocity": "cart velocity",
    "abs_cart_velocity": "|cart velocity|",
    "pole_angle": "pole angle",
    "abs_pole_angle": "|pole angle|",
    "sin_theta": "sin(theta)",
    "cos_theta": "cos(theta)",
    "pole_angular_velocity": "pole angular velocity",
    "abs_pole_angular_velocity": "|pole angular velocity|",
    "action_force": "cart force",
    "abs_action_force": "|cart force|",
    "fell": "fell",
    "animal_distance": "distance to animal",
    "abs_animal_distance": "|distance to animal|",
    "near_animal": "cart near animal",
    "pole_distance_to_animal": "pole distance to animal",
    "near_pole_touch": "pole near animal",
    "cart_hit_animal": "cart hit animal",
    "pole_hit_animal": "pole hit animal",
    "hit_animal": "cart or pole hit animal",
    "target_cart_position_error": "distance from target position",
    "target_pole_angle_error": "distance from target angle",
}


DEFAULT_REWARD_TERMS: tuple[dict[str, Any], ...] = ()


DEFAULT_REWARD_WEIGHTS: dict[str, Any] = {
    "reward_terms": [dict(term) for term in DEFAULT_REWARD_TERMS],
}


BASELINE_REWARD_WEIGHTS: dict[str, Any] = {
    "reward_terms": [{"signal": "alive", "factor": 1.0, "scale": "unit"}],
}


OBSERVATION_DEMO_REWARD_WEIGHTS: dict[str, Any] = {
    "reward_terms": [
        {"signal": "alive", "factor": 1.0, "scale": "unit"},
        {"signal": "pole_angle", "factor": 1.0, "transform": "cos", "scale": "pi"},
        {"signal": "pole_angle", "factor": -2.0, "transform": "abs", "scale": "pi"},
        {"signal": "pole_angular_velocity", "factor": -0.2, "transform": "abs", "scale": "unit"},
        {"signal": "cart_position", "factor": -0.1, "transform": "abs", "scale": "unit"},
        {"signal": "fell", "factor": -8.0, "scale": "unit"},
    ],
}


REWARD_DEMO_WEIGHTS: dict[str, dict[str, Any]] = {
    # Reliable upright balance: reward standing straight, punish falling.
    "balance": {
        "reward_terms": [
            {"signal": "alive", "factor": 1.0, "scale": "unit"},
            {"signal": "pole_angle", "factor": 1.0, "transform": "cos", "scale": "pi"},
            {"signal": "pole_angle", "factor": -2.0, "transform": "abs", "scale": "pi"},
            {"signal": "fell", "factor": -8.0, "scale": "unit"},
        ],
    },
    # Go as fast as possible without driving off the screen.
    "max_cart_velocity": {
        "reward_terms": [
            {"signal": "cart_velocity", "factor": 1.0, "transform": "abs", "scale": "unit"},
            {"signal": "cart_off_screen", "factor": -10.0, "scale": "unit"},
        ],
    },
    # Spin the pole as fast as possible; falling is allowed (no fell penalty,
    # and the episode does not stop when the pole tips over).
    "max_pole_velocity": {
        "reward_terms": [
            {"signal": "pole_angular_velocity", "factor": 1.0, "transform": "abs", "scale": "unit"},
            {"signal": "cart_off_screen", "factor": -10.0, "scale": "unit"},
        ],
    },
}


CONTROLLED_DEMO_VERSION = "controlled-demo-v4-matched"
DEMO_EPISODES = 400
# The reward demos need to converge enough to show the cos-vs-sin difference, so
# they train longer than the snappy observation/action demos.
REWARD_DEMO_EPISODES = 800
DEMO_MAX_STEPS = 300
DEMO_LEARNING_RATE = 0.25
DEMO_GAMMA = 0.99
DEMO_EPSILON = 0.9
DEMO_EPSILON_MIN = 0.05
DEMO_Q_BINS = 6


REWARD_SCALE_LABELS: dict[str, str] = {
    "unit": "[-1,1]",
    "pi": "[-pi,pi]",
}


# Continuous observation signals that can be read either on their normalized
# [-1,1] scale or stretched onto [-pi,pi] (so cos/sin treat the value as an angle).
REWARD_SCALE_SIGNALS: set[str] = {
    "cart_position",
    "abs_cart_position",
    "cart_velocity",
    "abs_cart_velocity",
    "pole_angle",
    "abs_pole_angle",
    "pole_angular_velocity",
    "abs_pole_angular_velocity",
    "sin_theta",
    "cos_theta",
    "action_force",
    "abs_action_force",
    "animal_distance",
    "abs_animal_distance",
    "pole_distance_to_animal",
}


def reward_signal_display(signal: str, transform: str = "raw") -> str:
    if signal.startswith("abs_"):
        signal = signal.removeprefix("abs_")
        transform = "abs"

    label = REWARD_SIGNAL_LABELS.get(signal, signal)
    if transform == "abs":
        return f"|{label}|"
    if transform == "sin":
        return f"sin({label})"
    if transform == "cos":
        return f"cos({label})"
    return label


OBSERVATION_LABELS: dict[str, str] = {
    "cart_position": "Cart position",
    "cart_velocity": "Cart velocity",
    "pole_angle": "Pole angle",
    "sin_theta": "sin(theta)",
    "cos_theta": "cos(theta)",
    "pole_angular_velocity": "Pole angular velocity",
    "animal_distance": "Distance to animal",
    "abs_animal_distance": "|Distance to animal|",
}


OBSERVATION_DESCRIPTIONS: dict[str, str] = {
    "cart_position": "Where the cart is on the track.",
    "cart_velocity": "How fast the cart is moving left or right.",
    "pole_angle": "Which way the pole is leaning.",
    "pole_angular_velocity": "How fast the pole is rotating.",
    "sin_theta": "A smooth angle helper based on pole angle.",
    "cos_theta": "A smooth uprightness helper based on pole angle.",
    "animal_distance": "Signed distance from the cart to the animal.",
}


FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "cart_position": (-1.0, 1.0),
    "cart_velocity": (-1.0, 1.0),
    "pole_angle": (-1.0, 1.0),
    "sin_theta": (-1.0, 1.0),
    "cos_theta": (-1.0, 1.0),
    "pole_angular_velocity": (-1.0, 1.0),
    "animal_distance": (-1.0, 1.0),
    "abs_animal_distance": (0.0, 1.0),
}


DEFAULT_OBSERVATION_FEATURES: tuple[str, ...] = (
    "cart_position",
    "cart_velocity",
    "pole_angle",
    "pole_angular_velocity",
)


_DRAG_CANVAS_COMPONENT: Any | None = None


def drag_canvas_component(
    *,
    mode: str,
    title: str,
    pool: list[dict[str, Any]],
    value: list[Any],
    key: str,
    height: int,
    reset_id: str = "",
) -> Any:
    """Render the local drag/drop builder and return its JSON value."""
    global _DRAG_CANVAS_COMPONENT
    components = require_dependencies("streamlit.components.v1")["streamlit.components.v1"]
    if _DRAG_CANVAS_COMPONENT is None:
        component_dir = Path(__file__).parent / "drag_canvas"
        _DRAG_CANVAS_COMPONENT = components.declare_component(
            "drag_canvas",
            path=str(component_dir),
        )

    return _DRAG_CANVAS_COMPONENT(
        mode=mode,
        title=title,
        pool=pool,
        value=value,
        height=height,
        reset_id=reset_id,
        default=value,
        key=key,
    )


ACTION_PRESETS: dict[str, tuple[float, ...]] = {
    "Standard left/right": (-10.0, 10.0),
    "Gentle left/none/right": (-5.0, 0.0, 5.0),
    "Strong left/none/right": (-15.0, 0.0, 15.0),
    "Five force levels": (-15.0, -7.5, 0.0, 7.5, 15.0),
    "Custom": (-10.0, 10.0),
}


TUTORIAL_STEPS: tuple[dict[str, str], ...] = (
    {
        "target": "training",
        "title": "Training controls",
        "body": "Pick the learning method, practice budget, learning speed, and exploration. In RL, exploration decides how often the agent tries actions before trusting its current policy.",
    },
    {
        "target": "ethical",
        "title": "Ethical exploration",
        "body": "This changes the environment. The animal creates a safety constraint, and optional distance/contact signals let students ask what the agent can know and what it should value.",
    },
    {
        "target": "observation_action",
        "title": "Observations and actions",
        "body": "Observations are the state inputs the policy receives. Actions are the force choices it can take. The policy learns a mapping from observations to actions.",
    },
    {
        "target": "reward",
        "title": "Reward function",
        "body": "Reward is the learning signal. Dragged blocks define what the agent treats as success, pain, safety, or progress at each environment step.",
    },
    {
        "target": "train",
        "title": "Train agent",
        "body": "Training repeatedly collects transitions: observation, action, reward, next observation. Q-learning updates a table; DQN updates a neural network from replay samples.",
    },
    {
        "target": "replay",
        "title": "Replay",
        "body": "Replay runs the learned policy without random exploration so students can see what behavior the reward and observations produced.",
    },
    {
        "target": "curve",
        "title": "Learning curve",
        "body": "The curve shows returns over episodes. Rising CartPole score means the policy is keeping the pendulum balanced for longer.",
    },
    {
        "target": "policy",
        "title": "Policy map",
        "body": "This view opens the learned value function: which force the agent prefers, and how valuable it thinks each state is.",
    },
)


def require_dependencies(*module_names: str) -> dict[str, Any]:
    """Import external modules with a clear install message when missing."""
    modules: dict[str, Any] = {}
    missing: list[str] = []

    for module_name in module_names:
        try:
            modules[module_name] = importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)

    if missing:
        package_hint = ", ".join(sorted(missing))
        raise SystemExit(
            "Missing dependencies: "
            f"{package_hint}\n\n"
            "Create a virtual environment, activate it, then run:\n"
            "  python -m pip install -r requirements.txt"
        )

    return modules


@dataclass
class TrainSettings:
    algorithm: str
    episodes: int
    max_steps: int
    learning_rate: float
    gamma: float
    epsilon: float
    epsilon_min: float
    seed: int
    batch_size: int = 64
    hidden_size: int = 64
    target_update: int = 10
    update_every: int = 1
    parallel_envs: int = 1
    show_training_preview: bool = False
    training_preview_interval: int = 25
    q_bins_per_feature: int = 8
    observation_features: tuple[str, ...] = DEFAULT_OBSERVATION_FEATURES
    action_forces: tuple[float, ...] = ACTION_PRESETS["Standard left/right"]
    initial_state: tuple[float, float, float, float] | None = None
    terminate_on_angle: bool = True
    ethical_exploration: bool = False
    animal_position: float = 0.0
    animal_radius: float = 0.18
    animal_contact_ends_episode: bool = True
    # Half-pole length used by the CartPole physics (Gymnasium default is 0.5).
    pole_length: float = 0.5


@dataclass
class TrainingResult:
    algorithm: str
    returns: list[float]
    env_returns: list[float]
    episode_lengths: list[int]
    policy: Any
    settings: TrainSettings
    reward_weights: dict[str, Any]
    label: str = "Current reward"


def make_env(render: bool = False) -> Any:
    gymnasium = require_dependencies("gymnasium")["gymnasium"]
    render_mode = "rgb_array" if render else None
    return gymnasium.make("CartPole-v1", render_mode=render_mode)


def reset_env(env: Any, settings: TrainSettings, seed: int) -> Any:
    np = require_dependencies("numpy")["numpy"]
    obs, _ = env.reset(seed=seed)

    # Apply a custom pole length: a longer pole has more rotational inertia, so it
    # tips more slowly and is actually easier to balance.
    base_env = env.unwrapped
    base_env.length = float(settings.pole_length)
    base_env.polemass_length = base_env.masspole * base_env.length

    if settings.initial_state is None:
        return obs

    base_env.state = np.array(settings.initial_state, dtype=np.float64)
    return np.array(settings.initial_state, dtype=np.float32)


def step_cartpole_with_force(
    env: Any,
    force: float,
    terminate_on_angle: bool = True,
) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    """Step CartPole with an arbitrary horizontal force."""
    np = require_dependencies("numpy")["numpy"]
    base_env = env.unwrapped
    x, x_dot, theta, theta_dot = base_env.state

    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    temp = (force + base_env.polemass_length * np.square(theta_dot) * sintheta) / base_env.total_mass
    thetaacc = (base_env.gravity * sintheta - costheta * temp) / (
        base_env.length
        * (4.0 / 3.0 - base_env.masspole * np.square(costheta) / base_env.total_mass)
    )
    xacc = temp - base_env.polemass_length * thetaacc * costheta / base_env.total_mass

    if base_env.kinematics_integrator == "euler":
        x = x + base_env.tau * x_dot
        x_dot = x_dot + base_env.tau * xacc
        theta = theta + base_env.tau * theta_dot
        theta_dot = theta_dot + base_env.tau * thetaacc
    else:
        x_dot = x_dot + base_env.tau * xacc
        x = x + base_env.tau * x_dot
        theta_dot = theta_dot + base_env.tau * thetaacc
        theta = theta + base_env.tau * theta_dot

    base_env.state = np.array((x, x_dot, theta, theta_dot), dtype=np.float64)
    cart_out = x < -base_env.x_threshold or x > base_env.x_threshold
    angle_out = theta < -base_env.theta_threshold_radians or theta > base_env.theta_threshold_radians
    terminated = bool(cart_out or (terminate_on_angle and angle_out))
    return np.array(base_env.state, dtype=np.float32), 1.0, terminated, False, {}


def apply_animal_contact_termination(
    obs: Any,
    terminated: bool,
    settings: TrainSettings,
) -> bool:
    if not settings.ethical_exploration or not settings.animal_contact_ends_episode:
        return bool(terminated)

    contact = animal_contact(obs, settings.animal_position, settings.animal_radius)
    return bool(terminated or contact["hit"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    modules = require_dependencies("numpy")
    modules["numpy"].random.seed(seed)

    if importlib.util.find_spec("torch") is not None:
        torch = importlib.import_module("torch")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def epsilon_for_episode(settings: TrainSettings, episode: int) -> float:
    start = max(0.0, float(settings.epsilon))
    end = max(0.0, min(float(settings.epsilon_min), start))

    if start == 0.0:
        return 0.0

    if settings.episodes <= 1:
        return end

    fraction = episode / float(settings.episodes - 1)
    if end == 0.0:
        return max(0.0, start * (1.0 - fraction))

    return max(end, start * ((end / start) ** fraction))


def make_bins(features: tuple[str, ...], bins_per_feature: int) -> list[Any]:
    np = require_dependencies("numpy")["numpy"]
    return [
        np.linspace(FEATURE_RANGES[feature][0], FEATURE_RANGES[feature][1], bins_per_feature - 1)
        for feature in features
    ]


def discretize_state(obs: Any, bins: list[Any]) -> tuple[int, ...]:
    np = require_dependencies("numpy")["numpy"]
    state: list[int] = []

    for value, edges in zip(obs, bins):
        bucket = int(np.digitize(float(value), edges))
        state.append(int(np.clip(bucket, 0, len(edges))))

    return tuple(state)


def rolling_mean(values: list[float], window: int = 10) -> list[float]:
    if not values:
        return []

    smoothed: list[float] = []
    for index in range(len(values)):
        start = max(0, index + 1 - window)
        smoothed.append(sum(values[start : index + 1]) / (index + 1 - start))
    return smoothed


def checkpoint_episodes(total_episodes: int, count: int = 5) -> set[int]:
    """Episode numbers (1-indexed) at which to snapshot the policy: every 1/count."""
    total = max(1, int(total_episodes))
    count = max(1, min(count, total))
    return {max(1, round(total * (i + 1) / count)) for i in range(count)}


def train_q_learning(
    settings: TrainSettings,
    reward_weights: dict[str, Any],
    progress_callback: Callable[[int, int, list[float]], None] | None = None,
    label: str = "Current reward",
    checkpoint_callback: Callable[[int, int, TrainingResult], None] | None = None,
) -> TrainingResult:
    np = require_dependencies("numpy")["numpy"]
    seed_everything(settings.seed)

    env = make_env()
    env.action_space.seed(settings.seed)
    bins = make_bins(settings.observation_features, settings.q_bins_per_feature)
    q_shape = (
        tuple(settings.q_bins_per_feature for _ in settings.observation_features)
        + (len(settings.action_forces),)
    )
    q_table = np.zeros(q_shape, dtype=np.float32)
    checkpoints = checkpoint_episodes(settings.episodes) if checkpoint_callback else set()

    returns: list[float] = []
    env_returns: list[float] = []
    episode_lengths: list[int] = []

    for episode in range(settings.episodes):
        obs = reset_env(env, settings, settings.seed + episode)
        agent_obs = observation_function(
            obs,
            settings.observation_features,
            settings.animal_position,
        )
        state = discretize_state(agent_obs, bins)
        epsilon = epsilon_for_episode(settings, episode)
        total_reward = 0.0
        total_env_reward = 0.0

        for step in range(settings.max_steps):
            if random.random() < epsilon:
                action = random.randrange(len(settings.action_forces))
            else:
                action = int(np.argmax(q_table[state]))

            action_force = action_function(action, settings.action_forces)
            next_obs, env_reward, terminated, truncated, _ = step_cartpole_with_force(
                env,
                action_force,
                settings.terminate_on_angle,
            )
            terminated = apply_animal_contact_termination(next_obs, terminated, settings)
            done = bool(terminated or truncated)
            shaped_reward = reward_function(
                obs,
                action,
                action_force,
                next_obs,
                float(env_reward),
                bool(terminated),
                bool(truncated),
                reward_weights,
            )

            next_agent_obs = observation_function(
                next_obs,
                settings.observation_features,
                settings.animal_position,
            )
            next_state = discretize_state(next_agent_obs, bins)
            best_next_q = 0.0 if done else float(np.max(q_table[next_state]))
            target = shaped_reward + settings.gamma * best_next_q
            q_table[state + (action,)] += settings.learning_rate * (
                target - float(q_table[state + (action,)])
            )

            obs = next_obs
            state = next_state
            total_reward += shaped_reward
            total_env_reward += float(env_reward)

            if done:
                episode_lengths.append(step + 1)
                break
        else:
            episode_lengths.append(settings.max_steps)

        returns.append(total_reward)
        env_returns.append(total_env_reward)

        if progress_callback is not None:
            progress_callback(episode + 1, settings.episodes, list(returns))

        if checkpoint_callback is not None and (episode + 1) in checkpoints:
            snapshot = TrainingResult(
                algorithm="Q-learning",
                returns=list(returns),
                env_returns=list(env_returns),
                episode_lengths=list(episode_lengths),
                policy={"q_table": q_table.copy(), "bins": bins},
                settings=settings,
                reward_weights=dict(reward_weights),
                label=label,
            )
            checkpoint_callback(episode + 1, settings.episodes, snapshot)

    env.close()

    return TrainingResult(
        algorithm="Q-learning",
        returns=returns,
        env_returns=env_returns,
        episode_lengths=episode_lengths,
        policy={"q_table": q_table, "bins": bins},
        settings=settings,
        reward_weights=dict(reward_weights),
        label=label,
    )


class ReplayBuffer:
    def __init__(self, capacity: int = 2_000) -> None:
        self.items: deque[tuple[Any, int, float, Any, bool]] = deque(maxlen=capacity)

    def add(self, obs: Any, action: int, reward: float, next_obs: Any, done: bool) -> None:
        self.items.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> list[tuple[Any, int, float, Any, bool]]:
        return random.sample(self.items, batch_size)

    def __len__(self) -> int:
        return len(self.items)


def make_q_network(input_size: int, hidden_size: int, output_size: int) -> Any:
    torch_modules = require_dependencies("torch")
    torch = torch_modules["torch"]
    nn = torch.nn

    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )


def train_dqn(
    settings: TrainSettings,
    reward_weights: dict[str, Any],
    progress_callback: Callable[[int, int, list[float]], None] | None = None,
    label: str = "Current reward",
    checkpoint_callback: Callable[[int, int, TrainingResult], None] | None = None,
) -> TrainingResult:
    modules = require_dependencies("numpy", "torch")
    np = modules["numpy"]
    torch = modules["torch"]
    nn = torch.nn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))

    seed_everything(settings.seed)
    env_count = max(1, int(settings.parallel_envs))
    envs = [make_env() for _ in range(env_count)]
    for index, env in enumerate(envs):
        env.action_space.seed(settings.seed + index)
    checkpoints = checkpoint_episodes(settings.episodes) if checkpoint_callback else set()

    obs_size = len(settings.observation_features)
    action_size = len(settings.action_forces)

    model = make_q_network(obs_size, settings.hidden_size, action_size).to(device)
    target_model = make_q_network(obs_size, settings.hidden_size, action_size).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    loss_fn = nn.SmoothL1Loss()
    replay = ReplayBuffer()

    returns: list[float] = []
    env_returns: list[float] = []
    episode_lengths: list[int] = []
    update_count = 0
    global_step = 0

    observations = [
        reset_env(env, settings, settings.seed + env_index)
        for env_index, env in enumerate(envs)
    ]
    agent_observations = [
        observation_function(obs, settings.observation_features, settings.animal_position)
        for obs in observations
    ]
    running_returns = [0.0 for _ in envs]
    running_env_returns = [0.0 for _ in envs]
    running_lengths = [0 for _ in envs]

    def optimize_once() -> None:
        nonlocal update_count

        if len(replay) < settings.batch_size:
            return

        batch = replay.sample(settings.batch_size)
        obs_batch = torch.tensor(
            np.array([item[0] for item in batch]),
            dtype=torch.float32,
            device=device,
        )
        action_batch = torch.tensor(
            [item[1] for item in batch],
            dtype=torch.int64,
            device=device,
        ).unsqueeze(1)
        reward_batch = torch.tensor(
            [item[2] for item in batch],
            dtype=torch.float32,
            device=device,
        )
        next_obs_batch = torch.tensor(
            np.array([item[3] for item in batch]),
            dtype=torch.float32,
            device=device,
        )
        done_batch = torch.tensor(
            [item[4] for item in batch],
            dtype=torch.float32,
            device=device,
        )

        q_values = model(obs_batch).gather(1, action_batch).squeeze(1)
        with torch.no_grad():
            next_q_values = target_model(next_obs_batch).max(dim=1).values
            targets = reward_batch + settings.gamma * (1.0 - done_batch) * next_q_values

        loss = loss_fn(q_values, targets)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        update_count += 1

    while len(returns) < settings.episodes:
        epsilon = epsilon_for_episode(settings, len(returns))
        with torch.no_grad():
            obs_tensor = torch.tensor(
                np.array(agent_observations),
                dtype=torch.float32,
                device=device,
            )
            greedy_actions = (
                torch.argmax(model(obs_tensor), dim=1)
                .cpu()
                .numpy()
                .astype(int)
                .tolist()
            )

        for env_index, env in enumerate(envs):
            if len(returns) >= settings.episodes:
                break

            if random.random() < epsilon:
                action = random.randrange(action_size)
            else:
                action = greedy_actions[env_index]

            obs = observations[env_index]
            agent_obs = agent_observations[env_index]
            action_force = action_function(action, settings.action_forces)
            next_obs, env_reward, terminated, _, _ = step_cartpole_with_force(
                env,
                action_force,
                settings.terminate_on_angle,
            )
            terminated = apply_animal_contact_termination(next_obs, terminated, settings)
            running_lengths[env_index] += 1
            truncated = running_lengths[env_index] >= settings.max_steps
            done = bool(terminated or truncated)
            next_agent_obs = observation_function(
                next_obs,
                settings.observation_features,
                settings.animal_position,
            )
            shaped_reward = reward_function(
                obs,
                action,
                action_force,
                next_obs,
                float(env_reward),
                bool(terminated),
                bool(truncated),
                reward_weights,
            )
            replay.add(agent_obs, action, shaped_reward, next_agent_obs, done)

            running_returns[env_index] += shaped_reward
            running_env_returns[env_index] += float(env_reward)
            global_step += 1

            if global_step % settings.update_every == 0:
                optimize_once()

            if done:
                returns.append(running_returns[env_index])
                env_returns.append(running_env_returns[env_index])
                episode_lengths.append(running_lengths[env_index])

                if len(returns) % settings.target_update == 0:
                    target_model.load_state_dict(model.state_dict())

                if progress_callback is not None:
                    progress_callback(len(returns), settings.episodes, list(returns))

                if checkpoint_callback is not None and len(returns) in checkpoints:
                    snapshot = TrainingResult(
                        algorithm="DQN",
                        returns=list(returns),
                        env_returns=list(env_returns),
                        episode_lengths=list(episode_lengths),
                        policy={
                            "model": model,
                            "updates": update_count,
                            "parallel_envs": env_count,
                            "device": str(device),
                        },
                        settings=settings,
                        reward_weights=dict(reward_weights),
                        label=label,
                    )
                    checkpoint_callback(len(returns), settings.episodes, snapshot)

                reset_seed = settings.seed + env_count + len(returns) + env_index
                observations[env_index] = reset_env(env, settings, reset_seed)
                agent_observations[env_index] = observation_function(
                    observations[env_index],
                    settings.observation_features,
                    settings.animal_position,
                )
                running_returns[env_index] = 0.0
                running_env_returns[env_index] = 0.0
                running_lengths[env_index] = 0
            else:
                observations[env_index] = next_obs
                agent_observations[env_index] = next_agent_obs

        optimize_once()

    target_model.load_state_dict(model.state_dict())
    for env in envs:
        env.close()

    return TrainingResult(
        algorithm="DQN",
        returns=returns,
        env_returns=env_returns,
        episode_lengths=episode_lengths,
        policy={
            "model": model,
            "updates": update_count,
            "parallel_envs": env_count,
            "device": str(device),
        },
        settings=settings,
        reward_weights=dict(reward_weights),
        label=label,
    )


def train_agent(
    settings: TrainSettings,
    reward_weights: dict[str, Any],
    progress_callback: Callable[[int, int, list[float]], None] | None = None,
    label: str = "Current reward",
    checkpoint_callback: Callable[[int, int, TrainingResult], None] | None = None,
) -> TrainingResult:
    if settings.algorithm == "Q-learning":
        return train_q_learning(settings, reward_weights, progress_callback, label, checkpoint_callback)
    return train_dqn(settings, reward_weights, progress_callback, label, checkpoint_callback)


def choose_action_for_result(result: TrainingResult, obs: Any) -> int:
    np = require_dependencies("numpy")["numpy"]
    agent_obs = observation_function(
        obs,
        result.settings.observation_features,
        result.settings.animal_position,
    )

    if result.algorithm == "Q-learning":
        q_table = result.policy["q_table"]
        bins = result.policy["bins"]
        state = discretize_state(agent_obs, bins)
        return int(np.argmax(q_table[state]))

    torch = require_dependencies("torch")["torch"]
    model = result.policy["model"]
    model.eval()
    model_device = next(model.parameters()).device
    with torch.no_grad():
        obs_tensor = torch.tensor(agent_obs, dtype=torch.float32, device=model_device).unsqueeze(0)
        return int(torch.argmax(model(obs_tensor), dim=1).item())


def append_fall_animation(
    env: Any,
    frames: list[Any],
    frame_limit: int | None,
    max_fall_frames: int = 120,
) -> None:
    """Continue the visualization after failure without changing the episode score."""
    base_env = env.unwrapped
    x, _, theta, theta_dot = [float(value) for value in base_env.state]

    if abs(theta) < 0.02:
        direction = theta_dot if abs(theta_dot) > 0.001 else 1.0
        theta = math.copysign(0.02, direction)

    for _ in range(max_fall_frames):
        if frame_limit is not None and len(frames) >= frame_limit:
            break

        angular_acceleration = (base_env.gravity / base_env.length) * math.sin(theta)
        theta_dot = (theta_dot + base_env.tau * angular_acceleration) * 0.995
        theta += base_env.tau * theta_dot

        fall_state = base_env.state.copy()
        fall_state[:] = (x, 0.0, theta, theta_dot)
        base_env.state = fall_state
        frames.append(env.render())

        if abs(theta) >= math.pi * 0.95:
            break


def evaluate_policy(
    result: TrainingResult,
    seed: int,
    render: bool = False,
    sleep_limit: int | None = None,
) -> tuple[float, float, int, list[Any]]:
    env = make_env(render=render)
    obs = reset_env(env, result.settings, seed)
    total_reward = 0.0
    total_env_reward = 0.0
    frames: list[Any] = []

    for step in range(result.settings.max_steps):
        if render:
            frame = env.render()
            frames.append(frame)

        action = choose_action_for_result(result, obs)
        action_force = action_function(action, result.settings.action_forces)
        next_obs, env_reward, terminated, truncated, _ = step_cartpole_with_force(
            env,
            action_force,
            result.settings.terminate_on_angle,
        )
        terminated = apply_animal_contact_termination(next_obs, terminated, result.settings)
        shaped_reward = reward_function(
            obs,
            action,
            action_force,
            next_obs,
            float(env_reward),
            bool(terminated),
            bool(truncated),
            result.reward_weights,
        )
        total_reward += shaped_reward
        total_env_reward += float(env_reward)
        obs = next_obs

        if terminated:
            if render:
                append_fall_animation(env, frames, sleep_limit)
            env.close()
            return total_reward, total_env_reward, step + 1, frames

        if truncated:
            env.close()
            return total_reward, total_env_reward, step + 1, frames

        if sleep_limit is not None and len(frames) >= sleep_limit:
            break

    env.close()
    return total_reward, total_env_reward, result.settings.max_steps, frames


def make_learning_curve(results: list[TrainingResult]) -> Any:
    modules = require_dependencies("matplotlib.pyplot")
    plt = modules["matplotlib.pyplot"]

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    for result in results:
        episodes = list(range(1, len(result.returns) + 1))
        axes[0].plot(
            episodes,
            rolling_mean(result.returns),
            label=f"{result.label}: shaped",
            linewidth=2,
        )
        axes[1].plot(
            episodes,
            rolling_mean(result.env_returns),
            label=f"{result.label}: env",
            linewidth=2,
        )

    axes[0].set_title("Reward the agent learned from")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Rolling return")
    axes[1].set_title("CartPole score")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Rolling score")

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()

    figure.tight_layout()
    return figure


def policy_value_grid(
    result: TrainingResult,
    cart_position: float,
    cart_velocity: float,
    resolution: int = 81,
) -> tuple[Any, Any, Any, Any]:
    """Evaluate both actions across an angle/angular-velocity state slice."""
    np = require_dependencies("numpy")["numpy"]
    angles = np.linspace(-0.2095, 0.2095, resolution)
    angular_velocities = np.linspace(-3.5, 3.5, resolution)
    action_count = len(result.settings.action_forces)
    q_values = np.zeros((resolution, resolution, action_count), dtype=np.float32)

    if result.algorithm == "Q-learning":
        q_table = result.policy["q_table"]
        bins = result.policy["bins"]
        for row, angular_velocity in enumerate(angular_velocities):
            for column, angle in enumerate(angles):
                agent_obs = observation_function(
                    (cart_position, cart_velocity, angle, angular_velocity),
                    result.settings.observation_features,
                    result.settings.animal_position,
                )
                state = discretize_state(agent_obs, bins)
                q_values[row, column] = q_table[state]
    else:
        torch = require_dependencies("torch")["torch"]
        angle_grid, angular_velocity_grid = np.meshgrid(
            angles,
            angular_velocities,
        )
        states = np.column_stack(
            (
                np.full(angle_grid.size, cart_position),
                np.full(angle_grid.size, cart_velocity),
                angle_grid.ravel(),
                angular_velocity_grid.ravel(),
            )
        )
        agent_states = np.array(
            [
                observation_function(
                    state,
                    result.settings.observation_features,
                    result.settings.animal_position,
                )
                for state in states
            ],
            dtype=np.float32,
        )
        model = result.policy["model"]
        model.eval()
        with torch.no_grad():
            q_values = (
                model(torch.tensor(agent_states, dtype=torch.float32))
                .cpu()
                .numpy()
                .reshape(resolution, resolution, action_count)
            )

    best_action_indices = np.argmax(q_values, axis=2)
    action_forces = np.array(result.settings.action_forces, dtype=np.float32)
    preferred_forces = action_forces[best_action_indices]
    best_value = np.max(q_values, axis=2)
    return angles, angular_velocities, preferred_forces, best_value


def make_policy_value_figure(
    result: TrainingResult,
    cart_position: float,
    cart_velocity: float,
) -> Any:
    modules = require_dependencies("matplotlib.pyplot")
    np = require_dependencies("numpy")["numpy"]
    plt = modules["matplotlib.pyplot"]
    angles, angular_velocities, preferred_forces, best_value = policy_value_grid(
        result,
        cart_position,
        cart_velocity,
    )

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    extent = [
        math.degrees(float(angles[0])),
        math.degrees(float(angles[-1])),
        float(angular_velocities[0]),
        float(angular_velocities[-1]),
    ]
    force_limit = max(float(np.max(np.abs(result.settings.action_forces))), 0.001)

    action_image = axes[0].imshow(
        preferred_forces,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="coolwarm",
        vmin=-force_limit,
        vmax=force_limit,
        interpolation="nearest" if result.algorithm == "Q-learning" else "bilinear",
    )
    if float(np.min(preferred_forces)) < 0.0 < float(np.max(preferred_forces)):
        axes[0].contour(
            np.degrees(angles),
            angular_velocities,
            preferred_forces,
            levels=[0.0],
            colors="black",
            linewidths=1.0,
        )
    axes[0].set_title("Preferred force")
    axes[0].set_xlabel("Pole angle (degrees)")
    axes[0].set_ylabel("Pole angular velocity")
    action_colorbar = figure.colorbar(action_image, ax=axes[0])
    action_colorbar.set_label("Cart force")

    value_image = axes[1].imshow(
        best_value,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        interpolation="nearest" if result.algorithm == "Q-learning" else "bilinear",
    )
    axes[1].set_title("Best predicted value")
    axes[1].set_xlabel("Pole angle (degrees)")
    axes[1].set_ylabel("Pole angular velocity")
    value_colorbar = figure.colorbar(value_image, ax=axes[1])
    value_colorbar.set_label("max Q(state, action)")

    figure.suptitle(
        f"{result.algorithm} at cart x={cart_position:.2f}, velocity={cart_velocity:.2f}"
    )
    figure.tight_layout()
    return figure


def summarize_result(result: TrainingResult) -> dict[str, float | int | str]:
    tail = max(1, min(20, len(result.returns)))
    return {
        "Run": result.label,
        "Algorithm": result.algorithm,
        "Episodes": len(result.returns),
        "Mean shaped reward": round(sum(result.returns[-tail:]) / tail, 2),
        "Mean CartPole score": round(sum(result.env_returns[-tail:]) / tail, 2),
        "Best CartPole score": round(max(result.env_returns), 2),
        "Mean episode length": round(sum(result.episode_lengths[-tail:]) / tail, 2),
    }


def run_smoke_test() -> None:
    print("Running Q-learning smoke test...")
    q_settings = TrainSettings(
        algorithm="Q-learning",
        episodes=4,
        max_steps=60,
        learning_rate=0.2,
        gamma=0.95,
        epsilon=0.7,
        epsilon_min=0.05,
        seed=4,
        q_bins_per_feature=5,
    )
    q_result = train_q_learning(q_settings, DEFAULT_REWARD_WEIGHTS)
    q_eval = evaluate_policy(q_result, seed=104, render=False)

    print("Running DQN smoke test...")
    dqn_settings = TrainSettings(
        algorithm="DQN",
        episodes=4,
        max_steps=60,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=0.8,
        epsilon_min=0.1,
        seed=8,
        batch_size=16,
        hidden_size=32,
        target_update=2,
        update_every=2,
        parallel_envs=2,
    )
    dqn_result = train_dqn(dqn_settings, DEFAULT_REWARD_WEIGHTS)
    dqn_eval = evaluate_policy(dqn_result, seed=108, render=False)

    print(
        "Smoke test passed.\n"
        f"  Q-learning episodes: {len(q_result.returns)}, eval score: {q_eval[1]:.0f}\n"
        f"  DQN episodes: {len(dqn_result.returns)}, eval score: {dqn_eval[1]:.0f}"
    )


def tutorial_enabled(st: Any) -> bool:
    return bool(st.session_state.get("tutorial_enabled", False))


def tutorial_step_index(st: Any) -> int:
    step = int(st.session_state.get("tutorial_step", 0))
    step = max(0, min(step, len(TUTORIAL_STEPS) - 1))
    st.session_state["tutorial_step"] = step
    return step


def active_tutorial_step(st: Any) -> dict[str, str] | None:
    if not tutorial_enabled(st):
        return None
    return TUTORIAL_STEPS[tutorial_step_index(st)]


def tutorial_target_for_step(index: int) -> str:
    index = max(0, min(index, len(TUTORIAL_STEPS) - 1))
    return TUTORIAL_STEPS[index]["target"]


def request_streamlit_rerun(st: Any) -> None:
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun is not None:
        rerun()


def set_app_stage(st: Any, stage: str) -> None:
    st.session_state["app_stage"] = stage
    request_streamlit_rerun(st)


def render_intro_page(st: Any) -> None:
    gif_bytes = intro_demo_gif(st)
    if gif_bytes:
        encoded = base64.b64encode(gif_bytes).decode("ascii")
        background = (
            "background-image: linear-gradient(90deg, rgba(15, 23, 42, 0.86), "
            "rgba(15, 23, 42, 0.32)), "
            f"url(data:image/gif;base64,{encoded});"
        )
    else:
        background = "background: linear-gradient(120deg, #172033, #285d62);"

    st.markdown(
        f"""
        <style>
        .intro-hero {{
            {background}
            background-size: cover;
            background-position: center;
            min-height: 68vh;
            border-radius: 8px;
            display: flex;
            align-items: flex-end;
            padding: clamp(1.4rem, 4vw, 3.5rem);
            color: white;
        }}
        .intro-copy {{
            max-width: 720px;
        }}
        .intro-copy h1 {{
            font-size: clamp(2.2rem, 6vw, 5rem);
            line-height: 0.96;
            margin: 0 0 0.8rem 0;
            letter-spacing: 0;
        }}
        .intro-copy p {{
            font-size: clamp(1rem, 1.6vw, 1.25rem);
            line-height: 1.45;
            margin: 0;
            max-width: 620px;
        }}
        .intro-start-button + div[data-testid="stButton"] > button {{
            min-height: 4.25rem;
            font-size: 1.15rem;
            font-weight: 800;
            border-radius: 8px;
        }}
        </style>
        <div class="intro-hero">
            <div class="intro-copy">
                <h1>Live RL Pendulum Lab</h1>
                <p>Use reinforcement learning to balance an inverted pendulum, then change the learning problem yourself.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, tutorial_column, skip_column, right = st.columns([0.7, 1.2, 1.2, 0.7])
    with tutorial_column:
        st.markdown('<span class="intro-start-button"></span>', unsafe_allow_html=True)
        if st.button("Start tutorial", type="primary", use_container_width=True):
            st.session_state["tutorial_enabled"] = True
            st.session_state["tutorial_step"] = 0
            set_app_stage(st, "background")
    with skip_column:
        st.markdown('<span class="intro-start-button"></span>', unsafe_allow_html=True)
        if st.button("Skip to activity", use_container_width=True):
            st.session_state["tutorial_enabled"] = False
            set_app_stage(st, "lab")


def render_background_page(st: Any) -> None:
    st.title("Background")
    st.markdown(
        """
        The inverted pendulum is a common platform used for research in controls and machine learning.
        The problem consists of figuring out how to make the pendulum balance itself for as long as possible.

        We are going to learn how to use reinforcement learning to balance the pendulum as our introduction to RL.

        Reinforcement learning is used to solve elaborate problems with edge cases by allowing algorithms to learn how to solve them on their own.
        For instance, walking is very difficult to describe fully in code, so we let algorithms learn how to figure it out on their own in simulation.
        """
    )
    st.subheader("Three Things To Design")
    columns = st.columns(3)
    with columns[0]:
        st.markdown("**Observations**")
        st.write("What your agent can see when interacting with the environment.")
    with columns[1]:
        st.markdown("**Actions**")
        st.write("What your agent can do based on what it sees.")
    with columns[2]:
        st.markdown("**Reward Function**")
        st.write("How you teach your agent what to do based on what it sees.")

    st.caption("We will go over each one in depth.")
    left, center, right = st.columns([1, 1.2, 1])
    with center:
        if st.button("Continue", type="primary", use_container_width=True):
            set_app_stage(st, "observation_demo")


# ---------------------------------------------------------------------------
# Precomputed demo assets: the fixed (non-interactive) slide demos are trained
# once via `--precompute` and saved to disk so the slide pages load instantly.
# ---------------------------------------------------------------------------
DEMO_ASSETS_PATH = Path(__file__).resolve().parent / "assets" / "demo_assets.pkl"


def figure_to_png_bytes(figure: Any) -> bytes:
    """Render a matplotlib figure to PNG bytes so it can be cached on disk."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    return buffer.getvalue()


def load_demo_assets(st: Any) -> dict[str, Any]:
    """Load the precomputed demo asset bundle from disk once per session."""
    if "demo_assets" in st.session_state:
        return st.session_state["demo_assets"]
    assets: dict[str, Any] = {}
    if DEMO_ASSETS_PATH.exists():
        try:
            with open(DEMO_ASSETS_PATH, "rb") as handle:
                assets = pickle.load(handle)
        except Exception:
            assets = {}
    st.session_state["demo_assets"] = assets
    return assets


def build_observation_demo_run(
    *,
    label: str,
    features: tuple[str, ...],
    seed: int,
    initial_state: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> dict[str, Any]:
    settings = TrainSettings(
        algorithm="Q-learning",
        episodes=DEMO_EPISODES,
        max_steps=DEMO_MAX_STEPS,
        learning_rate=DEMO_LEARNING_RATE,
        gamma=DEMO_GAMMA,
        epsilon=DEMO_EPSILON,
        epsilon_min=DEMO_EPSILON_MIN,
        seed=seed,
        q_bins_per_feature=DEMO_Q_BINS,
        observation_features=features,
        initial_state=initial_state,
    )
    result = train_q_learning(settings, OBSERVATION_DEMO_REWARD_WEIGHTS, label=label)
    shaped, env_score, length, frames = evaluate_policy(
        result,
        seed=seed + 200,
        render=True,
        sleep_limit=180,
    )
    return {
        "label": label,
        "features": features,
        "score": env_score,
        "length": length,
        "shaped": shaped,
        "gif_bytes": frames_to_gif(frames, fps=30),
    }


def _build_observation_demo() -> dict[str, Any]:
    return {
        "full": build_observation_demo_run(
            label="Full observations",
            features=DEFAULT_OBSERVATION_FEATURES,
            seed=31,
        ),
        "limited": build_observation_demo_run(
            label="No pole angle or pole spin",
            features=("cart_position", "cart_velocity"),
            seed=31,
            # Start slightly tilted. Without pole angle/spin observations the
            # agent cannot tell which way to correct, so it fails more visibly.
            initial_state=(0.0, 0.0, 0.06, 0.0),
        ),
    }


def observation_demo_cache(st: Any) -> dict[str, Any]:
    assets = load_demo_assets(st)
    if "observation_demo" in assets:
        return dict(assets["observation_demo"])
    if "observation_demo_cache" not in st.session_state:
        st.session_state["observation_demo_cache"] = _build_observation_demo()
    return dict(st.session_state["observation_demo_cache"])


def build_controlled_demo_run(
    *,
    features: tuple[str, ...],
    action_forces: tuple[float, ...],
    seed: int,
) -> dict[str, Any]:
    settings = TrainSettings(
        algorithm="Q-learning",
        episodes=DEMO_EPISODES,
        max_steps=DEMO_MAX_STEPS,
        learning_rate=DEMO_LEARNING_RATE,
        gamma=DEMO_GAMMA,
        epsilon=DEMO_EPSILON,
        epsilon_min=DEMO_EPSILON_MIN,
        seed=seed,
        q_bins_per_feature=DEMO_Q_BINS,
        observation_features=features,
        action_forces=action_forces,
        # Always start the pole perfectly upright and centered for the interactive
        # demos so students see a consistent starting state, not a random tilt.
        initial_state=(0.0, 0.0, 0.0, 0.0),
    )
    result = train_q_learning(settings, OBSERVATION_DEMO_REWARD_WEIGHTS, label="Demo")
    _, env_score, _, frames = evaluate_policy(
        result,
        seed=seed + 200,
        render=True,
        sleep_limit=180,
    )
    return {
        "version": CONTROLLED_DEMO_VERSION,
        "features": features,
        "action_forces": action_forces,
        "score": env_score,
        "gif_bytes": frames_to_gif(frames, fps=30),
    }


def cached_controlled_demo_run(
    st: Any,
    *,
    cache_name: str,
    features: tuple[str, ...],
    action_forces: tuple[float, ...],
    seed: int,
) -> dict[str, Any]:
    cache = st.session_state.setdefault(cache_name, {})
    key = (CONTROLLED_DEMO_VERSION, features, action_forces, seed)
    if key not in cache:
        cache[key] = build_controlled_demo_run(
            features=features,
            action_forces=action_forces,
            seed=seed,
        )
    return dict(cache[key])


def render_observation_slideshow_page(st: Any) -> None:
    st.markdown(
        """
        <style>
        .observation-slide-lead {
            font-size: 1.28rem;
            line-height: 1.55;
            margin: 0.35rem 0 1rem;
            color: #243447;
        }
        .observation-slide-note {
            font-size: 1.18rem;
            line-height: 1.55;
            margin: 1rem 0 1.25rem;
            color: #344054;
        }
        .observation-slide-features {
            font-size: 1.05rem;
            line-height: 1.45;
            margin: 0 0 0.8rem;
            color: #475467;
        }
        .observation-slide-features.demo-caption {
            min-height: 4.2rem;
        }
        .lab-snippet {
            border: 1px solid #d0d5dd;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0 1.25rem;
            background: #f9fafb;
        }
        .lab-snippet-title {
            font-size: 1.15rem;
            font-weight: 800;
            color: #182230;
            margin-bottom: 0.75rem;
        }
        .lab-snippet-row {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 0.8rem;
            align-items: center;
        }
        .lab-snippet-panel {
            min-height: 6rem;
            border: 1px dashed #98a2b3;
            border-radius: 8px;
            padding: 0.85rem;
            background: white;
        }
        .lab-snippet-label {
            font-size: 0.95rem;
            font-weight: 800;
            color: #475467;
            margin-bottom: 0.55rem;
        }
        .lab-chip {
            display: inline-block;
            border: 1px solid #98a2b3;
            border-radius: 999px;
            padding: 0.42rem 0.7rem;
            margin: 0.18rem;
            background: #ffffff;
            color: #182230;
            font-weight: 700;
            font-size: 0.95rem;
        }
        .lab-chip.selected {
            border-color: #2563eb;
            background: #eff6ff;
            color: #1d4ed8;
        }
        .snippet-arrow {
            color: #667085;
            font-size: 1.8rem;
            font-weight: 800;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Observation Demo")
    st.markdown(
        """
        <div class="observation-slide-note">
            Observations are what the agent can see. If the agent cannot observe
            the pole angle or how fast the pole is rotating, it has to act from
            incomplete information.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Training two small Q-learning agents for the observation demo..."):
        demo = observation_demo_cache(st)

    full_column, limited_column = st.columns(2)
    for column, key in ((full_column, "full"), (limited_column, "limited")):
        run = demo[key]
        with column:
            st.subheader(run["label"])
            features = ", ".join(OBSERVATION_LABELS[feature] for feature in run["features"])
            st.markdown(
                f'<div class="observation-slide-features demo-caption"><strong>Observations:</strong> {features}</div>',
                unsafe_allow_html=True,
            )
            if run["gif_bytes"]:
                st.image(run["gif_bytes"], width="stretch")

    st.markdown(
        """
        <div class="lab-snippet">
            <div class="lab-snippet-title">In the lab, students change observations by dragging them into the agent's observation box.</div>
            <div class="lab-snippet-row">
                <div class="lab-snippet-panel">
                    <div class="lab-snippet-label">Observation pool</div>
                    <span class="lab-chip">Cart position</span>
                    <span class="lab-chip">Cart velocity</span>
                    <span class="lab-chip">Pole angle</span>
                    <span class="lab-chip">Pole angular velocity</span>
                </div>
                <div class="snippet-arrow">→</div>
                <div class="lab-snippet-panel">
                    <div class="lab-snippet-label">Agent sees</div>
                    <span class="lab-chip selected">Cart position</span>
                    <span class="lab-chip selected">Pole angle</span>
                </div>
            </div>
        </div>
        <div class="observation-slide-lead">
            <strong>Both agents are trained the same way:</strong> same reward,
            same Q-learning setup, same episode budget. Only the observations change.
        </div>
        <div class="observation-slide-note">
            Removing pole angle and pole spin usually changes the behavior
            dramatically because those observations tell the agent which way the
            pendulum is falling.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Try changing observations")
    st.markdown(
        '<div class="observation-slide-note">Use the same fixed Q-learning setup and predesigned reward. Only change what the agent can see.</div>',
        unsafe_allow_html=True,
    )
    observation_pool = [
        {"id": "cart_position", "label": OBSERVATION_LABELS["cart_position"], "group": "Cart"},
        {"id": "cart_velocity", "label": OBSERVATION_LABELS["cart_velocity"], "group": "Cart"},
        {"id": "pole_angle", "label": OBSERVATION_LABELS["pole_angle"], "group": "Pole"},
        {
            "id": "pole_angular_velocity",
            "label": OBSERVATION_LABELS["pole_angular_velocity"],
            "group": "Pole",
        },
    ]
    st.session_state.setdefault("observation_demo_builder_features", list(DEFAULT_OBSERVATION_FEATURES))
    selected_features = list(st.session_state["observation_demo_builder_features"])
    playground_columns = st.columns([1, 1])
    with playground_columns[0]:
        component_value = drag_canvas_component(
            mode="observation",
            title="Agent observations",
            pool=observation_pool,
            value=selected_features,
            key="observation_demo_drag_canvas",
            height=330,
            reset_id="observation-demo",
        )
        if isinstance(component_value, list):
            selected_features = [
                str(feature)
                for feature in component_value
                if str(feature) in OBSERVATION_LABELS
            ]
            if not selected_features:
                selected_features = ["pole_angle"]
            st.session_state["observation_demo_builder_features"] = selected_features
        train_observation_demo = st.button("Train with these observations", type="primary", use_container_width=True)

    with playground_columns[1]:
        if train_observation_demo:
            with st.spinner("Training controlled observation demo..."):
                st.session_state["observation_demo_playground_run"] = cached_controlled_demo_run(
                    st,
                    cache_name="controlled_observation_demo_cache",
                    features=tuple(selected_features),
                    action_forces=ACTION_PRESETS["Standard left/right"],
                    seed=21,
                )
        run = st.session_state.get("observation_demo_playground_run")
        selected_feature_tuple = tuple(selected_features)
        if (
            isinstance(run, dict)
            and run.get("gif_bytes")
            and run.get("version") == CONTROLLED_DEMO_VERSION
            and tuple(run.get("features", ())) == selected_feature_tuple
        ):
            shown_features = ", ".join(OBSERVATION_LABELS[feature] for feature in run["features"])
            st.markdown(
                f'<div class="observation-slide-features"><strong>Agent sees:</strong> {shown_features}</div>',
                unsafe_allow_html=True,
            )
            st.image(run["gif_bytes"], width="stretch")
        else:
            st.info("Train once to see this observation choice in action.")

    can_continue = isinstance(st.session_state.get("observation_demo_playground_run"), dict)
    back_column, next_column = st.columns(2)
    if back_column.button("Back", use_container_width=True):
        set_app_stage(st, "background")
    if can_continue and next_column.button("Continue", type="primary", use_container_width=True, key="observation_demo_continue"):
        set_app_stage(st, "action_demo")


def build_action_demo_run(
    *,
    label: str,
    action_forces: tuple[float, ...],
    description: str,
    seed: int,
    terminate_on_angle: bool = True,
) -> dict[str, Any]:
    settings = TrainSettings(
        algorithm="Q-learning",
        episodes=DEMO_EPISODES,
        max_steps=DEMO_MAX_STEPS,
        learning_rate=DEMO_LEARNING_RATE,
        gamma=DEMO_GAMMA,
        epsilon=DEMO_EPSILON,
        epsilon_min=DEMO_EPSILON_MIN,
        seed=seed,
        q_bins_per_feature=DEMO_Q_BINS,
        observation_features=DEFAULT_OBSERVATION_FEATURES,
        action_forces=action_forces,
        terminate_on_angle=terminate_on_angle,
        initial_state=(0.0, 0.0, 0.0, 0.0),
    )
    result = train_q_learning(settings, OBSERVATION_DEMO_REWARD_WEIGHTS, label=label)
    _, _, _, frames = evaluate_policy(
        result,
        seed=seed + 300,
        render=True,
        sleep_limit=300,
    )
    return {
        "label": label,
        "description": description,
        "action_forces": action_forces,
        "gif_bytes": frames_to_gif(frames, fps=30),
    }


def _build_action_demo() -> dict[str, Any]:
    return {
        "strong": build_action_demo_run(
            label="Strong left or strong right",
            action_forces=(-15.0, 15.0),
            description="Only two big pushes, no gentle option. Every correction is a hard shove, so the cart jerks back and forth and overshoots instead of settling — it cannot hold the pole steady.",
            seed=31,
            terminate_on_angle=False,
        ),
        "gentle": build_action_demo_run(
            label="Gentle left, coast, gentle right",
            action_forces=(-5.0, 0.0, 5.0),
            description="Smaller pushes plus a no-force coast action. With fine control the agent balances the pole and keeps it upright.",
            seed=31,
        ),
        "right_biased": build_action_demo_run(
            label="One left, several right",
            action_forces=(-10.0, 0.0, 5.0, 10.0, 15.0),
            description="Most of the force options push right, so the agent is biased that way and drifts the cart rightward across the track instead of balancing.",
            seed=11,
            terminate_on_angle=False,
        ),
    }


def action_demo_cache(st: Any) -> dict[str, Any]:
    assets = load_demo_assets(st)
    if "action_demo" in assets:
        return dict(assets["action_demo"])
    if "action_demo_cache" not in st.session_state:
        st.session_state["action_demo_cache"] = _build_action_demo()
    return dict(st.session_state["action_demo_cache"])


ACTION_DEMO_CHOICES: dict[str, tuple[float, ...]] = {
    "Strong left/right": (-15.0, 15.0),
    "Gentle left/none/right": (-5.0, 0.0, 5.0),
    "One left, several right": (-10.0, 0.0, 5.0, 10.0, 15.0),
    "Five force levels": (-15.0, -7.5, 0.0, 7.5, 15.0),
}


def seed_for_action_demo(action_forces: tuple[float, ...]) -> int:
    rounded_forces = tuple(round(float(force), 3) for force in action_forces)
    stable_seeds = {
        (-10.0, 10.0): 21,
        (-15.0, 15.0): 31,
        (-5.0, 0.0, 5.0): 31,
        (-10.0, 0.0, 5.0, 10.0, 15.0): 31,
        (-15.0, -7.5, 0.0, 7.5, 15.0): 42,
    }
    return stable_seeds.get(rounded_forces, 7)


def build_reward_demo_run(
    *,
    label: str,
    reward_weights: dict[str, Any],
    formula: str,
    description: str,
    seed: int,
    terminate_on_angle: bool = True,
) -> dict[str, Any]:
    settings = TrainSettings(
        algorithm="Q-learning",
        episodes=REWARD_DEMO_EPISODES,
        max_steps=DEMO_MAX_STEPS,
        learning_rate=DEMO_LEARNING_RATE,
        gamma=DEMO_GAMMA,
        epsilon=DEMO_EPSILON,
        epsilon_min=DEMO_EPSILON_MIN,
        seed=seed,
        q_bins_per_feature=DEMO_Q_BINS,
        observation_features=DEFAULT_OBSERVATION_FEATURES,
        action_forces=ACTION_PRESETS["Standard left/right"],
        terminate_on_angle=terminate_on_angle,
        initial_state=(0.0, 0.0, 0.0, 0.0),
    )
    result = train_q_learning(settings, reward_weights, label=label)
    _, env_score, _, frames = evaluate_policy(
        result,
        seed=seed + 200,
        render=True,
        sleep_limit=180,
    )
    return {
        "label": label,
        "formula": formula,
        "description": description,
        "score": env_score,
        "gif_bytes": frames_to_gif(frames, fps=30),
    }


def _build_reward_demo() -> dict[str, Any]:
    return {
        "balance": build_reward_demo_run(
            label="Balance the pole",
            reward_weights=REWARD_DEMO_WEIGHTS["balance"],
            formula="reward = alive + cos(pole angle) - 2 x |pole angle| - 8 x fell",
            description="The classic goal. Rewarding staying alive and an upright pole, and punishing falling, gives a policy that reliably balances.",
            seed=41,
        ),
        "max_pole_velocity": build_reward_demo_run(
            label="Spin the pole",
            reward_weights=REWARD_DEMO_WEIGHTS["max_pole_velocity"],
            formula="reward = |pole angular velocity| - 10 x cart off screen",
            description="Reward only how fast the pole spins, with no penalty for falling and no stoppage when it tips. The agent spins the pole around instead of balancing it.",
            seed=33,
            terminate_on_angle=False,
        ),
    }


def reward_demo_cache(st: Any) -> dict[str, Any]:
    assets = load_demo_assets(st)
    if "reward_demo" in assets:
        return dict(assets["reward_demo"])
    # Tie the cache to the episode budget so retraining at a new length rebuilds.
    cache_key = f"reward_demo_cache_e{REWARD_DEMO_EPISODES}_v3"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = _build_reward_demo()
    return dict(st.session_state[cache_key])


def render_reward_slideshow_page(st: Any) -> None:
    st.markdown(
        """
        <style>
        .reward-slide-note {
            font-size: 1.18rem;
            line-height: 1.55;
            margin: 1rem 0 1.25rem;
            color: #344054;
        }
        .reward-formula {
            border: 1px solid #d0d5dd;
            border-radius: 8px;
            padding: 0.8rem;
            margin: 0.7rem 0 0.9rem;
            background: #f9fafb;
            color: #182230;
            font-size: 1.05rem;
            font-weight: 800;
            min-height: 6.2rem;
        }
        .reward-slide-note.demo-caption {
            min-height: 5.4rem;
        }
        .reward-slide-lead {
            font-size: 1.28rem;
            line-height: 1.55;
            margin: 1rem 0 1.25rem;
            color: #243447;
        }
        .reward-teach-card {
            border: 1px solid #d0d5dd;
            border-radius: 10px;
            padding: 1rem 1.15rem;
            margin: 0.9rem 0;
            background: #ffffff;
        }
        .reward-teach-card h4 {
            margin: 0 0 0.55rem;
            font-size: 1.12rem;
            color: #182230;
        }
        .reward-teach-card p {
            font-size: 1.05rem;
            line-height: 1.5;
            color: #344054;
            margin: 0.4rem 0;
        }
        .reward-inline-code {
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            background: #f2f4f7;
            border-radius: 5px;
            padding: 0.1rem 0.38rem;
            color: #1d4ed8;
            font-weight: 700;
            font-size: 0.96rem;
        }
        .reward-compare {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.8rem;
            margin: 0.6rem 0 0.2rem;
        }
        .reward-compare-cell {
            border: 1px solid #e4e7ec;
            border-radius: 8px;
            padding: 0.7rem 0.85rem;
            background: #f9fafb;
        }
        .reward-compare-cell strong {
            color: #182230;
        }
        .reward-table {
            width: 100%;
            border-collapse: collapse;
            margin: 0.5rem 0 0.2rem;
            font-size: 0.98rem;
        }
        .reward-table th, .reward-table td {
            border: 1px solid #e4e7ec;
            padding: 0.42rem 0.6rem;
            text-align: left;
            color: #344054;
        }
        .reward-table th {
            background: #f2f4f7;
            color: #182230;
        }
        .reward-pos { color: #027a48; font-weight: 700; }
        .reward-neg { color: #b42318; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Reward Demo")
    st.markdown(
        """
        <div class="reward-slide-note">
            The reward function is how you teach the agent what behavior is worth
            repeating. The same simple signals can encode completely different goals:
            balance the pole, or spin the pole as fast as possible. Each policy below
            was trained with only the building blocks from the pool.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Training two small Q-learning agents for the reward demo..."):
        demo = reward_demo_cache(st)

    columns = st.columns(2)
    for column, key in zip(columns, ("balance", "max_pole_velocity")):
        run = demo[key]
        with column:
            st.subheader(run["label"])
            st.markdown(f'<div class="reward-formula">{run["formula"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="reward-slide-note demo-caption">{run["description"]}</div>', unsafe_allow_html=True)
            if run["gif_bytes"]:
                st.image(run["gif_bytes"], width="stretch")

    st.subheader("How the math inside a reward block works")
    st.markdown(
        """
        <div class="reward-teach-card">
            <h4>1. Everything is measured on a fixed scale first</h4>
            <p>
                Before the agent sees a number, the lab squeezes it into a
                predictable range. Positions and velocities are normalized to
                <span class="reward-inline-code">[-1, 1]</span>, and the pole angle
                is wrapped into <span class="reward-inline-code">[-&pi;, &pi;]</span>
                so that &quot;straight up&quot; is 0 and a full lean to either side is
                &plusmn;&pi;. Keeping every signal on the same scale means one term
                cannot accidentally drown out the others.
            </p>
        </div>
        <div class="reward-teach-card">
            <h4>2. <span class="reward-inline-code">cos</span> and <span class="reward-inline-code">sin</span> point the agent at different goals</h4>
            <p>
                Once a signal is mapped to the range
                <span class="reward-inline-code">-&pi;</span> to
                <span class="reward-inline-code">&pi;</span> (0 in the middle), a math
                function turns that distance into a reward, and the function you pick
                decides <em>which</em> value pays the most.
            </p>
            <div class="reward-compare">
                <div class="reward-compare-cell">
                    <strong>cos(distance)</strong> peaks at <strong>0</strong>.<br>
                    cos(0) = 1 (best), cos(&plusmn;&pi;) = -1 (worst). Use it to say
                    &quot;reward being right in the middle.&quot;
                </div>
                <div class="reward-compare-cell">
                    <strong>sin(distance)</strong> peaks at <strong>&pi;/2</strong>.<br>
                    sin(0) = 0, sin(&plusmn;&pi;) = 0, sin(&pi;/2) = 1. Use it to say
                    &quot;reward being off to one side,&quot; not centered.
                </div>
            </div>
            <table class="reward-table">
                <tr><th>distance (-&pi;..&pi;)</th><th>cos</th><th>sin</th></tr>
                <tr><td>0 (centered)</td><td class="reward-pos">1.0 (max)</td><td>0.0</td></tr>
                <tr><td>&pi;/2 (off to one side)</td><td>0.0</td><td class="reward-pos">1.0 (max)</td></tr>
                <tr><td>&plusmn;&pi; (far edge)</td><td class="reward-neg">-1.0 (min)</td><td>0.0</td></tr>
            </table>
            <p>
                So <span class="reward-inline-code">cos(distance from -&pi; to &pi;)</span>
                rewards getting that distance to <strong>0</strong> (the center), while
                <span class="reward-inline-code">sin(distance from -&pi; to &pi;)</span>
                rewards being <strong>off to one side</strong> instead. Same input, very
                different behavior.
            </p>
        </div>
        <div class="reward-teach-card">
            <h4>3. Absolute value turns a direction into a distance</h4>
            <p>
                A raw signal like cart position is signed: -0.6 means left of center,
                +0.6 means right of center. Wrapping it in
                <span class="reward-inline-code">|cart position|</span> throws away the
                direction and keeps only &quot;how far off.&quot; Now left and right are
                treated the same, so a term like
                <span class="reward-inline-code">-0.3 &times; |cart position|</span>
                punishes drifting <em>either</em> way and pulls the cart back toward the
                middle.
            </p>
        </div>
        <div class="reward-teach-card">
            <h4>4. The sign of the number is the instruction</h4>
            <p>
                A <span class="reward-pos">big positive</span> reward tells the agent
                &quot;do more of this.&quot; A <span class="reward-neg">big negative</span>
                reward tells it &quot;stop doing this.&quot; The size sets how strongly it
                cares: <span class="reward-inline-code">-8 &times; fell</span> is a loud
                &quot;never fall,&quot; while <span class="reward-inline-code">-0.1 &times; |cart position|</span>
                is a gentle nudge. Zero means &quot;I don't care about this signal.&quot;
            </p>
        </div>
        <div class="reward-teach-card">
            <h4>5. Reward signals you can combine</h4>
            <p>These are the building blocks available in the reward block:</p>
            <table class="reward-table">
                <tr><th>signal</th><th>what it measures</th></tr>
                <tr><td>alive</td><td>+1 for every step the episode survives</td></tr>
                <tr><td>fell</td><td>1 on the step the pole falls (pair with a negative factor)</td></tr>
                <tr><td>pole angle</td><td>tilt of the pole, wrapped to [-&pi;, &pi;]</td></tr>
                <tr><td>cart position / velocity</td><td>where the cart is and how fast it moves</td></tr>
                <tr><td>pole angular velocity</td><td>how fast the pole is rotating</td></tr>
                <tr><td>distance from target position</td><td>how far the cart is from a target spot (0 = on target)</td></tr>
                <tr><td>distance from target angle</td><td>how far the pole is from a target lean (0 = on target)</td></tr>
            </table>
            <p>
                Any of these can be transformed with
                <span class="reward-inline-code">cos</span>,
                <span class="reward-inline-code">sin</span>, or
                <span class="reward-inline-code">abs</span>, then scaled by a positive
                or negative factor. The two demos above use exactly these pieces
                &mdash; rewarding an upright pole gives balancing, while
                <span class="reward-inline-code">|pole angular velocity|</span> with no
                fall penalty gives a spinning pole.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="reward-slide-lead">
            Next, start the tutorial and use the full pendulum lab. You will get
            observations, actions, reward blocks, starting states, and ethical
            exploration all together.
        </div>
        """,
        unsafe_allow_html=True,
    )
    back_column, next_column = st.columns(2)
    if back_column.button("Back", use_container_width=True):
        set_app_stage(st, "action_demo")
    if next_column.button("Continue", type="primary", use_container_width=True):
        set_app_stage(st, "algorithm_demo")


ALGO_DEMO_VERSION = "algo-demo-v1"
ALGO_DEMO_REWARD_WEIGHTS: dict[str, Any] = {
    "reward_terms": [
        {"signal": "alive", "factor": 1.0, "scale": "unit"},
        {"signal": "pole_angle", "factor": 1.0, "transform": "cos", "scale": "pi"},
        {"signal": "pole_angle", "factor": -2.0, "transform": "abs", "scale": "pi"},
        {"signal": "cart_position", "factor": -0.1, "transform": "abs", "scale": "unit"},
        {"signal": "fell", "factor": -8.0, "scale": "unit"},
    ],
}


def _build_algorithm_demo_assets() -> dict[str, Any]:
    """Train the Q-table/DQN demo and return only the picklable render data."""
    payload = _train_algorithm_demo()
    return {
        "q_map_png": figure_to_png_bytes(payload["q_map"]),
        "dqn_map_png": figure_to_png_bytes(payload["dqn_map"]),
        "q_table_shape": payload["q_table_shape"],
        "q_table_cells": payload["q_table_cells"],
    }


def build_algorithm_demo(st: Any) -> dict[str, Any]:
    """Train one small Q-table and one small DQN on the same balance task."""
    cache_key = f"algorithm_demo_cache_{ALGO_DEMO_VERSION}"
    if cache_key in st.session_state:
        return dict(st.session_state[cache_key])
    return _train_algorithm_demo(st, cache_key)


def _train_algorithm_demo(st: Any | None = None, cache_key: str | None = None) -> dict[str, Any]:

    common = dict(
        max_steps=DEMO_MAX_STEPS,
        learning_rate=DEMO_LEARNING_RATE,
        gamma=DEMO_GAMMA,
        epsilon=DEMO_EPSILON,
        epsilon_min=DEMO_EPSILON_MIN,
        seed=7,
        observation_features=DEFAULT_OBSERVATION_FEATURES,
        action_forces=ACTION_PRESETS["Standard left/right"],
    )

    q_settings = TrainSettings(
        algorithm="Q-learning",
        episodes=600,
        q_bins_per_feature=8,
        **common,
    )
    dqn_settings = TrainSettings(
        algorithm="DQN",
        episodes=120,
        hidden_size=64,
        learning_rate=0.001,
        batch_size=64,
        target_update=10,
        **{key: value for key, value in common.items() if key != "learning_rate"},
    )

    q_result = train_q_learning(q_settings, ALGO_DEMO_REWARD_WEIGHTS, label="Q-table")
    dqn_result = train_dqn(dqn_settings, ALGO_DEMO_REWARD_WEIGHTS, label="DQN")

    payload = {
        "q_result": q_result,
        "dqn_result": dqn_result,
        "q_map": make_policy_value_figure(q_result, cart_position=0.0, cart_velocity=0.0),
        "dqn_map": make_policy_value_figure(dqn_result, cart_position=0.0, cart_velocity=0.0),
        "q_table_shape": q_result.policy["q_table"].shape,
        "q_table_cells": int(q_result.policy["q_table"].size),
    }
    if st is not None and cache_key is not None:
        st.session_state[cache_key] = payload
    return dict(payload)


def render_algorithm_demo_page(st: Any) -> None:
    st.markdown(
        """
        <style>
        .algo-note {
            font-size: 1.18rem;
            line-height: 1.55;
            margin: 1rem 0 1.25rem;
            color: #344054;
        }
        .algo-lead {
            font-size: 1.28rem;
            line-height: 1.55;
            margin: 0.35rem 0 1rem;
            color: #243447;
        }
        .algo-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin: 0.6rem 0 0.4rem;
        }
        .algo-card {
            border: 1px solid #d0d5dd;
            border-radius: 10px;
            padding: 1rem 1.15rem;
            background: #ffffff;
        }
        .algo-card h4 {
            margin: 0 0 0.5rem;
            font-size: 1.16rem;
            color: #182230;
        }
        .algo-card .algo-tag {
            display: inline-block;
            font-size: 0.85rem;
            font-weight: 800;
            border-radius: 999px;
            padding: 0.15rem 0.6rem;
            margin-bottom: 0.55rem;
        }
        .algo-tag.fast { background: #ecfdf3; color: #027a48; }
        .algo-tag.slow { background: #fef3f2; color: #b42318; }
        .algo-card ul { margin: 0.4rem 0 0; padding-left: 1.1rem; }
        .algo-card li { font-size: 1.02rem; line-height: 1.5; color: #344054; margin: 0.25rem 0; }
        .algo-diagram {
            border: 1px dashed #98a2b3;
            border-radius: 8px;
            padding: 0.85rem;
            margin: 0.6rem 0 0.2rem;
            background: #f9fafb;
            text-align: center;
        }
        .qcell-grid {
            display: inline-grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 3px;
        }
        .qcell {
            width: 20px; height: 20px;
            border-radius: 3px;
            background: #d1e9ff;
            border: 1px solid #b2ddff;
        }
        .qcell.hot { background: #2e90fa; border-color: #1570cd; }
        .net-layer {
            display: inline-flex;
            flex-direction: column;
            gap: 4px;
            margin: 0 0.55rem;
            vertical-align: middle;
        }
        .net-node {
            width: 16px; height: 16px;
            border-radius: 50%;
            background: #fdb022;
            border: 1px solid #dc6803;
            margin: 0 auto;
        }
        .net-arrow { color: #667085; font-weight: 800; font-size: 1.4rem; vertical-align: middle; }
        .net-label { font-size: 0.8rem; color: #475467; margin-top: 0.3rem; }
        .algo-callout {
            border-left: 4px solid #2e90fa;
            background: #eff8ff;
            border-radius: 6px;
            padding: 0.85rem 1rem;
            margin: 1rem 0;
            font-size: 1.08rem;
            line-height: 1.5;
            color: #1849a9;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Q-table vs DQN")
    st.markdown(
        """
        <div class="algo-note">
            Both methods learn the same thing &mdash; how good each action is in each
            state &mdash; but they store that knowledge very differently. That single
            choice changes how fast they train and how smoothly they generalize.
        </div>
        """,
        unsafe_allow_html=True,
    )

    q_in = len(DEFAULT_OBSERVATION_FEATURES)
    q_out = len(ACTION_PRESETS["Standard left/right"])
    st.markdown(
        f"""
        <div class="algo-grid">
            <div class="algo-card">
                <span class="algo-tag fast">DISCRETE &middot; FAST</span>
                <h4>Q-table</h4>
                <div class="algo-diagram">
                    <div class="qcell-grid">
                        {''.join('<div class="qcell' + (' hot' if i in (8, 9, 14, 15, 20, 21) else '') + '"></div>' for i in range(36))}
                    </div>
                    <div class="net-label">a grid of cells, one stored number per state &times; action</div>
                </div>
                <ul>
                    <li>Chops each observation into bins, then stores a value in every cell.</li>
                    <li>Updating is just editing one cell &mdash; extremely fast and stable.</li>
                    <li>Learning in one cell does <em>not</em> help neighboring cells, so the map looks blocky.</li>
                    <li>Cells grow explosively as you add observations (the &quot;curse of dimensionality&quot;).</li>
                </ul>
            </div>
            <div class="algo-card">
                <span class="algo-tag slow">CONTINUOUS &middot; SLOWER</span>
                <h4>DQN (neural network)</h4>
                <div class="algo-diagram">
                    <span class="net-layer"><span class="net-node"></span><span class="net-node"></span><span class="net-node"></span><span class="net-node"></span><div class="net-label">{q_in} inputs</div></span>
                    <span class="net-arrow">&rarr;</span>
                    <span class="net-layer"><span class="net-node"></span><span class="net-node"></span><span class="net-node"></span><span class="net-node"></span><span class="net-node"></span><span class="net-node"></span><div class="net-label">64 hidden</div></span>
                    <span class="net-arrow">&rarr;</span>
                    <span class="net-layer"><span class="net-node"></span><span class="net-node"></span><span class="net-node"></span><span class="net-node"></span><span class="net-node"></span><span class="net-node"></span><div class="net-label">64 hidden</div></span>
                    <span class="net-arrow">&rarr;</span>
                    <span class="net-layer"><span class="net-node"></span><span class="net-node"></span><div class="net-label">{q_out} actions</div></span>
                </div>
                <ul>
                    <li>A small network reads the raw continuous state and predicts each action's value.</li>
                    <li>It generalizes &mdash; nearby states share weights, so the map is smooth.</li>
                    <li>Training is noisier: gradients, replay, and a target network take many more steps.</li>
                    <li>Scales to large/continuous observations where a table would be impossible.</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("The policy each one learned")
    st.markdown(
        """
        <div class="algo-note">
            These maps show the preferred push (left vs right) and how valuable the
            state is, sliced across pole angle and angular velocity. Watch the
            texture: the Q-table is blocky because each bin is independent, while the
            DQN is smooth because the network interpolates between states.
        </div>
        """,
        unsafe_allow_html=True,
    )

    assets = load_demo_assets(st)
    precomputed = assets.get("algorithm_demo")
    if precomputed:
        demo = precomputed
    else:
        with st.spinner("Training a Q-table and a DQN on the same balance task (the DQN takes longer)..."):
            demo = build_algorithm_demo(st)

    map_columns = st.columns(2)
    with map_columns[0]:
        st.markdown(
            f'<div class="algo-note"><strong>Q-table</strong> &mdash; {" &times; ".join(str(s) for s in demo["q_table_shape"])} = {demo["q_table_cells"]:,} stored cells, trained 600 episodes.</div>',
            unsafe_allow_html=True,
        )
        if "q_map_png" in demo:
            st.image(demo["q_map_png"], width="stretch")
        else:
            st.pyplot(demo["q_map"])
    with map_columns[1]:
        st.markdown(
            '<div class="algo-note"><strong>DQN</strong> &mdash; a 64-unit network, trained only 120 episodes and still smoothing in.</div>',
            unsafe_allow_html=True,
        )
        if "dqn_map_png" in demo:
            st.image(demo["dqn_map_png"], width="stretch")
        else:
            st.pyplot(demo["dqn_map"])

    st.markdown(
        """
        <div class="algo-callout">
            <strong>Where to start:</strong> begin with the <strong>Q-table</strong>.
            It trains in seconds, is easy to reason about, and almost always converges
            on CartPole. Once you understand the loop, switch to <strong>DQN</strong> to
            see how a network handles richer observations &mdash; but expect it to take
            longer, need more episodes, and be harder to get to a good policy.
        </div>
        """,
        unsafe_allow_html=True,
    )

    back_column, next_column = st.columns(2)
    if back_column.button("Back", use_container_width=True):
        set_app_stage(st, "reward_demo")
    if next_column.button("Continue to lab", type="primary", use_container_width=True):
        set_app_stage(st, "lab")


def render_action_slideshow_page(st: Any) -> None:
    st.markdown(
        """
        <style>
        .action-slide-lead {
            font-size: 1.28rem;
            line-height: 1.55;
            margin: 0.35rem 0 1rem;
            color: #243447;
        }
        .action-slide-note {
            font-size: 1.18rem;
            line-height: 1.55;
            margin: 1rem 0 1.25rem;
            color: #344054;
        }
        .action-slide-forces {
            font-size: 1.05rem;
            line-height: 1.45;
            margin: 0.25rem 0 0.8rem;
            color: #475467;
        }
        .action-slide-note.demo-caption {
            min-height: 5.4rem;
        }
        .action-slide-forces.demo-forces {
            min-height: 2.6rem;
        }
        .lab-snippet {
            border: 1px solid #d0d5dd;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0 1.25rem;
            background: #f9fafb;
        }
        .lab-snippet-title {
            font-size: 1.15rem;
            font-weight: 800;
            color: #182230;
            margin-bottom: 0.75rem;
        }
        .lab-action-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.8rem;
        }
        .lab-action-card {
            border: 1px solid #d0d5dd;
            border-radius: 8px;
            padding: 0.85rem;
            background: white;
        }
        .lab-action-card strong {
            display: block;
            font-size: 1rem;
            margin-bottom: 0.45rem;
            color: #182230;
        }
        .lab-chip {
            display: inline-block;
            border: 1px solid #98a2b3;
            border-radius: 999px;
            padding: 0.42rem 0.7rem;
            margin: 0.18rem;
            background: #ffffff;
            color: #182230;
            font-weight: 700;
            font-size: 0.95rem;
        }
        .lab-chip.selected {
            border-color: #2563eb;
            background: #eff6ff;
            color: #1d4ed8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Action Demo")
    st.markdown(
        """
        <div class="action-slide-note">
            Actions are what the agent is allowed to do. In CartPole, each action
            is a horizontal force on the cart. Changing the action menu changes
            the behaviors the agent can discover.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Training three small Q-learning agents for the action demo..."):
        demo = action_demo_cache(st)

    columns = st.columns(3)
    for column, key in zip(columns, ("strong", "gentle", "right_biased")):
        run = demo[key]
        with column:
            st.subheader(run["label"])
            st.markdown(
                f'<div class="action-slide-note demo-caption">{run["description"]}</div>',
                unsafe_allow_html=True,
            )
            forces = ", ".join(f"{force:g}" for force in run["action_forces"])
            st.markdown(
                f'<div class="action-slide-forces demo-forces"><strong>Force choices:</strong> {forces}</div>',
                unsafe_allow_html=True,
            )
            if run["gif_bytes"]:
                st.image(run["gif_bytes"], width="stretch")

    st.markdown(
        """
        <div class="lab-snippet">
            <div class="lab-snippet-title">In the lab, students change actions by choosing the force menu the policy can use.</div>
            <div class="lab-action-row">
                <div class="lab-action-card">
                    <strong>Two choices</strong>
                    <span class="lab-chip selected">-10</span>
                    <span class="lab-chip selected">+10</span>
                </div>
                <div class="lab-action-card">
                    <strong>Add coast</strong>
                    <span class="lab-chip selected">-5</span>
                    <span class="lab-chip selected">0</span>
                    <span class="lab-chip selected">+5</span>
                </div>
                <div class="lab-action-card">
                    <strong>More force levels</strong>
                    <span class="lab-chip selected">-15</span>
                    <span class="lab-chip selected">-7.5</span>
                    <span class="lab-chip selected">0</span>
                    <span class="lab-chip selected">+7.5</span>
                    <span class="lab-chip selected">+15</span>
                </div>
            </div>
        </div>
        <div class="action-slide-lead">
            <strong>All three agents are trained the same way:</strong> same
            observations, same reward, same Q-learning setup, same episode budget.
            Only the available actions change.
        </div>
        <div class="action-slide-note">
            A tiny action space can be easy to learn but coarse. A larger action
            space can be more expressive but gives the agent more choices to test.
            A no-force action lets the policy decide when doing nothing is useful.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Try changing actions")
    st.markdown(
        '<div class="action-slide-note">Use the same fixed Q-learning setup and predesigned reward. Only change the force choices the agent can use. Negative force pushes left, positive force pushes right, and 0 means no push.</div>',
        unsafe_allow_html=True,
    )
    playground_columns = st.columns([1, 1])
    with playground_columns[0]:
        st.session_state.setdefault(
            "action_demo_builder_items",
            action_forces_to_builder_items(ACTION_PRESETS["Standard left/right"]),
        )
        component_value = drag_canvas_component(
            mode="action",
            title="Agent actions",
            pool=action_builder_pool(),
            value=list(st.session_state["action_demo_builder_items"]),
            key="action_demo_drag_canvas",
            height=330,
            reset_id="action-demo-builder",
        )
        if isinstance(component_value, list):
            st.session_state["action_demo_builder_items"] = component_value
        action_forces = normalize_action_builder_items(st.session_state["action_demo_builder_items"])
        forces = ", ".join(f"{force:g}" for force in action_forces)
        st.markdown(
            f'<div class="action-slide-forces"><strong>Selected force choices:</strong> {forces}</div>',
            unsafe_allow_html=True,
        )
        train_action_demo = st.button("Train with these actions", type="primary", use_container_width=True)

    with playground_columns[1]:
        if train_action_demo:
            with st.spinner("Training controlled action demo..."):
                st.session_state["action_demo_playground_run"] = cached_controlled_demo_run(
                    st,
                    cache_name="controlled_action_demo_cache",
                    features=DEFAULT_OBSERVATION_FEATURES,
                    action_forces=action_forces,
                    seed=seed_for_action_demo(action_forces),
                )
        run = st.session_state.get("action_demo_playground_run")
        selected_action_tuple = tuple(action_forces)
        if (
            isinstance(run, dict)
            and run.get("gif_bytes")
            and run.get("version") == CONTROLLED_DEMO_VERSION
            and tuple(run.get("action_forces", ())) == selected_action_tuple
        ):
            shown_forces = ", ".join(f"{force:g}" for force in run["action_forces"])
            st.markdown(
                f'<div class="action-slide-forces"><strong>Agent actions:</strong> {shown_forces}</div>',
                unsafe_allow_html=True,
            )
            st.image(run["gif_bytes"], width="stretch")
        else:
            st.info("Train once to see this action space in action.")

    can_continue = isinstance(st.session_state.get("action_demo_playground_run"), dict)
    back_column, next_column = st.columns(2)
    if back_column.button("Back", use_container_width=True):
        set_app_stage(st, "observation_demo")
    if can_continue and next_column.button("Continue", type="primary", use_container_width=True, key="action_demo_continue"):
        set_app_stage(st, "reward_demo")


def render_tutorial_choice_page(st: Any) -> None:
    st.title("Choose Your Path")
    st.caption("Start with a guided walk-through, or jump straight into the lab.")

    start_column, skip_column = st.columns(2)
    with start_column:
        st.subheader("Start tutorial")
        st.write("Move through the page section by section and connect each control to the RL loop.")
        if st.button("Start tutorial", type="primary", use_container_width=True):
            st.session_state["tutorial_enabled"] = True
            st.session_state["tutorial_step"] = 0
            set_app_stage(st, "lab")
    with skip_column:
        st.subheader("Skip tutorial")
        st.write("Open the full activity immediately. You can turn the tutorial on later at the top.")
        if st.button("Skip", use_container_width=True):
            st.session_state["tutorial_enabled"] = False
            set_app_stage(st, "lab")


def render_tutorial_anchor(st: Any, target: str) -> None:
    st.markdown(f'<span id="tutorial-{target}"></span>', unsafe_allow_html=True)
    step = active_tutorial_step(st)
    pending_target = str(st.session_state.get("tutorial_scroll_target", ""))
    should_scroll = (step is not None and step["target"] == target) or pending_target == target
    if not should_scroll:
        return
    if pending_target == target:
        st.session_state.pop("tutorial_scroll_target", None)

    scroll_to_element(st, f"tutorial-{target}")


def scroll_to_element(st: Any, element_id: str) -> None:
    st.html(
        f"""
        <script>
        function scrollToTarget() {{
          const target = document.getElementById("{element_id}");
          if (target) {{
            target.scrollIntoView({{behavior: "smooth", block: "center"}});
          }}
        }}
        setTimeout(scrollToTarget, 80);
        setTimeout(scrollToTarget, 350);
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def sync_tutorial_results_view(st: Any, has_results: bool) -> None:
    if not has_results:
        return

    step = active_tutorial_step(st)
    if step is None:
        return

    if step["target"] == "replay":
        st.session_state["results_view"] = "Replay"
    elif step["target"] == "curve":
        st.session_state["results_view"] = "Results"
    elif step["target"] == "policy":
        st.session_state["results_view"] = "Policy map"


def render_tutorial_styles(st: Any) -> None:
    # The yellow tutorial callout boxes were removed, so there are no styles to inject.
    return


def render_tutorial_callout(st: Any, target: str, ui: Any | None = None) -> None:
    # Tutorial callouts (the yellow hint boxes) have been removed.
    return


def is_network_url_session(st: Any) -> bool:
    try:
        url = str(st.context.url or "")
    except Exception:
        return False

    if not url:
        return False

    host = url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0].strip("[]").lower()
    return host not in {"", "localhost", "127.0.0.1", "::1", "0.0.0.0"}


def sidebar_settings(st: Any, forced_algorithm: str | None = None) -> TrainSettings:
    render_tutorial_anchor(st, "training")
    st.sidebar.header("Training")
    if forced_algorithm is not None:
        algorithm = forced_algorithm
        st.sidebar.info(f"This mission requires: **{forced_algorithm}**")
    else:
        algorithm = st.sidebar.selectbox(
            "Algorithm",
            ["Q-learning", "DQN"],
            help="Q-learning stores action values in a table. DQN uses a small neural network to predict action values.",
        )
    cpu_count = os.cpu_count() or 1
    shared_hosting = is_network_url_session(st)
    default_episodes = 120 if algorithm == "Q-learning" or shared_hosting else 300
    max_episodes = 500 if algorithm == "Q-learning" else 3_000
    episode_step = 20 if algorithm == "Q-learning" else 50
    episodes = int(
        st.sidebar.slider(
            "Training episodes",
            20,
            max_episodes,
            default_episodes,
            episode_step,
            help="More episodes gives the agent more practice, but takes longer.",
        )
    )
    learning_rate = float(
        st.sidebar.slider(
            "Learning speed",
            0.01,
            1.0,
            0.2 if algorithm == "Q-learning" else 0.05,
            0.01,
            help="How strongly each new experience changes what the agent believes.",
        )
    )
    epsilon = float(
        st.sidebar.slider(
            "Exploration",
            0.0,
            1.0,
            0.8,
            0.05,
            help="Chance the agent tries a random action instead of its current best guess. High exploration means more trying things.",
        )
    )
    render_tutorial_callout(st, "training", st.sidebar)

    max_steps = 200 if algorithm == "DQN" and shared_hosting else 300
    gamma = 0.99
    epsilon_min = 0.05
    seed = 7
    q_bins_per_feature = 8
    target_update = 10
    update_every = 1
    parallel_envs = 1
    cpu_env_cap = max(1, cpu_count - 4)
    if shared_hosting:
        cpu_env_cap = 1
    dqn_parallel_options = [option for option in (1, 2, 4, 8, 12, 16) if option <= cpu_env_cap]
    if not dqn_parallel_options:
        dqn_parallel_options = [1]
    default_dqn_parallel_envs = 1 if shared_hosting else min(4, dqn_parallel_options[-1])

    if algorithm == "Q-learning":
        batch_size = 64
        hidden_size = 64
    else:
        learning_rate = learning_rate * 0.01
        batch_size = 16 if shared_hosting else 32
        hidden_size = 16 if shared_hosting else 32
        target_update = 5
        update_every = 10 if shared_hosting else 4
        parallel_envs = default_dqn_parallel_envs
        if shared_hosting:
            st.sidebar.info("Shared/Cloud mode: DQN uses 1 CPU env, 120 episodes, 200 max steps, smaller batches, fewer updates, and a smaller network.")

    with st.sidebar.expander("Advanced"):
        max_steps = int(st.slider("Max steps", 50, 500, max_steps, 25))
        gamma = float(st.slider("Future reward discount", 0.80, 0.999, gamma, 0.005))
        epsilon_min = float(st.slider("Final exploration", 0.0, 0.3, epsilon_min, 0.01))
        seed = int(st.number_input("Seed", min_value=0, max_value=999_999, value=seed))
        q_bins_per_feature = int(
            st.select_slider(
                "Q-table buckets",
                options=[4, 5, 6, 8, 10, 12],
                value=q_bins_per_feature,
            )
        )
        if algorithm == "DQN":
            batch_size_options = [16, 32] if shared_hosting else [16, 32, 64, 128]
            hidden_size_options = [16, 32] if shared_hosting else [16, 32, 64, 128]
            update_every_options = [8, 10, 12] if shared_hosting else [1, 2, 4, 8]
            batch_size = int(st.select_slider("DQN batch size", batch_size_options, batch_size))
            hidden_size = int(st.select_slider("DQN hidden units", hidden_size_options, hidden_size))
            update_every = int(st.select_slider("DQN update every N steps", update_every_options, update_every))
            target_update = int(st.select_slider("DQN target update episodes", [2, 5, 10, 20], target_update))
            parallel_envs = int(
                st.select_slider(
                    "DQN parallel CPU envs",
                    dqn_parallel_options,
                    parallel_envs,
                    help=(
                        f"Detected {cpu_count} CPU cores. "
                        "Network URL sessions are capped lower so multiple people can train at once."
                        if shared_hosting
                        else f"Detected {cpu_count} CPU cores. This keeps about 4 cores free when possible."
                    ),
                )
            )
    return TrainSettings(
        algorithm=algorithm,
        episodes=episodes,
        max_steps=max_steps,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        seed=seed,
        batch_size=batch_size,
        hidden_size=hidden_size,
        target_update=target_update,
        update_every=update_every,
        parallel_envs=parallel_envs,
        q_bins_per_feature=q_bins_per_feature,
    )


def normalize_reward_builder_items(items: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, float | str]]]:
    tokens: list[dict[str, Any]] = []
    terms: list[dict[str, float | str]] = []

    def normalize_items(raw_items: list[Any], collected_terms: list[dict[str, float | str]]) -> list[dict[str, Any]]:
        normalized_tokens: list[dict[str, Any]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue

            item_type = str(item.get("type", "term" if "signal" in item else ""))
            if item_type == "op":
                operator = "multiply" if item.get("op") in ("multiply", "*", "x") else "add"
                normalized_tokens.append({"type": "op", "op": operator})
                continue

            if item_type == "paren":
                value = "(" if item.get("value") == "(" else ")"
                normalized_tokens.append({"type": "paren", "value": value})
                continue

            if item_type == "func":
                function_name = str(item.get("func", "abs"))
                if function_name in ("abs", "sin", "cos", "min", "max"):
                    try:
                        factor = float(item.get("factor", 1.0))
                    except (TypeError, ValueError):
                        factor = 1.0
                    try:
                        threshold = float(item.get("threshold", 0.0))
                    except (TypeError, ValueError):
                        threshold = 0.0
                    normalized_tokens.append(
                        {
                            "type": "func",
                            "func": function_name,
                            "factor": factor,
                            "threshold": threshold,
                            "children": normalize_items(
                                item.get("children", []) if isinstance(item.get("children"), list) else [],
                                collected_terms,
                            ),
                        }
                    )
                continue

            if item_type != "term":
                continue

            signal = str(item.get("signal", "alive"))
            transform = str(item.get("transform", "abs" if item.get("absolute", False) else "raw"))
            connector = str(item.get("connector", "add"))
            scale = clean_reward_scale(signal, str(item.get("scale", default_reward_scale(signal))))
            try:
                factor = float(item.get("factor", 1.0))
            except (TypeError, ValueError):
                factor = 0.0

            if factor == 0.0 or signal not in REWARD_SIGNAL_LABELS:
                continue

            if item.get("type") != "term" and normalized_tokens:
                normalized_tokens.append(
                    {
                        "type": "op",
                        "op": "multiply" if connector == "multiply" else "add",
                    }
                )

            term_token = {
                "type": "term",
                "signal": signal,
                "factor": factor if transform == "raw" else 1.0,
                "scale": scale,
            }
            if transform in ("abs", "sin", "cos"):
                normalized_tokens.append(
                    {
                        "type": "func",
                        "func": transform,
                        "factor": factor,
                        "children": [term_token],
                    }
                )
            else:
                normalized_tokens.append(term_token)
            collected_terms.append(
                {
                    "signal": signal,
                    "factor": factor,
                    "transform": transform,
                    "connector": "multiply" if connector == "multiply" else "add",
                    "scale": scale,
                }
            )

        return normalized_tokens

    return normalize_items(items, terms), terms


def reward_controls(st: Any) -> dict[str, Any]:
    render_tutorial_anchor(st, "reward")
    st.subheader("Reward lab")
    st.caption("Drag signals and math blocks into the equation. min/max compare the nested value against the small cutoff box.")
    render_tutorial_callout(st, "reward")

    if "reward_builder_terms" not in st.session_state:
        st.session_state["reward_builder_terms"] = [dict(term) for term in DEFAULT_REWARD_TERMS]

    signal_groups: dict[str, list[str]] = {
        "Episode": [
            "alive",
            "fell",
        ],
        "Cart state": [
            "cart_position",
            "cart_velocity",
        ],
        "Track limits": [
            "cart_off_screen",
        ],
        "Pole state": [
            "pole_angle",
            "pole_angular_velocity",
        ],
        "Angle helpers": ["sin_theta", "cos_theta"],
        "Action": ["action_force"],
    }
    if st.session_state.get("ethical_exploration_enabled", False):
        signal_groups["Ethical exploration"] = [
            "hit_animal",
            "animal_distance",
            "near_animal",
            "pole_hit_animal",
            "pole_distance_to_animal",
            "near_pole_touch",
        ]

    reward_pool = [
        {
            "id": signal,
            "label": REWARD_SIGNAL_LABELS[signal],
            "group": group_name,
            "scales": REWARD_SCALE_LABELS if signal in REWARD_SCALE_SIGNALS else {},
            "default_scale": default_reward_scale(signal),
        }
        for group_name, signals in signal_groups.items()
        for signal in signals
    ]

    component_value = drag_canvas_component(
        mode="reward",
        title="Reward function",
        pool=reward_pool,
        value=list(st.session_state["reward_builder_terms"]),
        key="reward_drag_canvas",
        height=680,
        reset_id="reward-builder",
    )
    if isinstance(component_value, list):
        st.session_state["reward_builder_terms"] = component_value

    reward_tokens, reward_terms = normalize_reward_builder_items(
        list(st.session_state["reward_builder_terms"])
    )

    return {
        "reward_terms": reward_terms,
        "reward_tokens": reward_tokens,
    }


def parse_action_forces(text: str) -> tuple[float, ...]:
    forces = sorted({float(part.strip()) for part in text.split(",") if part.strip()})
    if len(forces) < 2:
        raise ValueError("Use at least two force values.")
    if any(not math.isfinite(force) for force in forces):
        raise ValueError("Forces must be finite numbers.")
    if any(abs(force) > 30.0 for force in forces):
        raise ValueError("Keep forces between -30 and 30 for a readable demo.")
    return tuple(forces)


def action_builder_pool() -> list[dict[str, Any]]:
    return [
        {"id": "force:left", "label": "Left push", "group": "Force bubbles", "default_force": -10.0},
        {"id": "force:none", "label": "No push", "group": "Force bubbles", "default_force": 0.0},
        {"id": "force:right", "label": "Right push", "group": "Force bubbles", "default_force": 10.0},
    ]


def normalize_action_builder_items(items: Any) -> tuple[float, ...]:
    if not isinstance(items, list):
        return ACTION_PRESETS["Standard left/right"]

    forces: list[float] = []
    for item in items:
        value = item.get("force") if isinstance(item, dict) else item
        try:
            force = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(force) and abs(force) <= 30.0:
            forces.append(force)

    unique_sorted = tuple(sorted(set(forces)))
    if len(unique_sorted) < 2:
        return ACTION_PRESETS["Standard left/right"]
    return unique_sorted


def action_forces_to_builder_items(action_forces: tuple[float, ...]) -> list[dict[str, float]]:
    return [{"force": float(force)} for force in action_forces]


def render_state_legend(st: Any) -> None:
    pd = require_dependencies("pandas")["pandas"]
    with st.expander("State legend", expanded=True):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Signal": "cart position",
                        "Zero means": "cart is centered",
                        "Positive means": "cart is right of center",
                    },
                    {
                        "Signal": "cart velocity",
                        "Zero means": "cart is not moving",
                        "Positive means": "cart moving right",
                    },
                    {
                        "Signal": "pole angle",
                        "Zero means": "pole is upright",
                        "Positive means": "pole leans right",
                    },
                    {
                        "Signal": "pole angular velocity",
                        "Zero means": "pole angle is not changing",
                        "Positive means": "pole rotating right",
                    },
                    {
                        "Signal": "cart force",
                        "Zero means": "no push",
                        "Positive means": "push right",
                    },
                    {
                        "Signal": "distance to animal",
                        "Zero means": "cart is at the animal",
                        "Positive means": "cart is right of animal",
                    },
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Agent observations are normalized before training. "
            "Each reward signal block can choose unit or pi scaling."
        )


def behavior_controls(
    st: Any,
    show_start_controls: bool = True,
    forced_start: str | None = None,
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, float, float, float] | None, bool]:
    render_tutorial_anchor(st, "observation_action")
    st.subheader("Observation and action lab")
    render_tutorial_callout(st, "observation_action")
    if show_start_controls:
        obs_column, action_column, start_column = st.columns(3)
    else:
        obs_column, action_column = st.columns(2)
        start_column = None

    with obs_column:
        st.markdown("**Observations**")
        st.caption("Drag observation bubbles into the canvas below. The agent only trains on the observations that land in that box.")
        if "observation_builder_features" not in st.session_state:
            st.session_state["observation_builder_features"] = list(DEFAULT_OBSERVATION_FEATURES)

        observation_groups = {
            "Cart": ["cart_position", "cart_velocity"],
            "Pole": ["pole_angle", "pole_angular_velocity"],
            "Angle helpers": ["sin_theta", "cos_theta"],
        }
        if st.session_state.get("ethical_exploration_enabled", False):
            observation_groups["Ethical exploration"] = ["animal_distance"]

        observation_pool = [
            {"id": feature, "label": OBSERVATION_LABELS[feature], "group": group_name}
            for group_name, features in observation_groups.items()
            for feature in features
        ]
        selected_features = list(st.session_state["observation_builder_features"])
        component_value = drag_canvas_component(
            mode="observation",
            title="Agent observations",
            pool=observation_pool,
            value=selected_features,
            key="observation_drag_canvas",
            height=390,
            reset_id=f"ethical:{st.session_state.get('ethical_exploration_enabled', False)}",
        )
        if isinstance(component_value, list):
            selected_features = [
                str(feature)
                for feature in component_value
                if str(feature) in OBSERVATION_LABELS
            ]
            st.session_state["observation_builder_features"] = selected_features

        if not selected_features:
            selected_features = ["pole_angle"]
            st.session_state["observation_builder_features"] = selected_features

        with st.expander("What the observation options mean", expanded=tutorial_enabled(st)):
            for feature in observation_pool:
                st.markdown(
                    f"**{feature['label']}**: {OBSERVATION_DESCRIPTIONS.get(feature['id'], 'An input the agent can use while choosing actions.')}"
                )

    with action_column:
        st.markdown("**Actions**")
        st.caption("Drag force bubbles into the canvas, then type the force value. Negative pushes left; positive pushes right; 0 means no push.")
        if "action_builder_items" not in st.session_state:
            st.session_state["action_builder_items"] = action_forces_to_builder_items(ACTION_PRESETS["Standard left/right"])

        component_value = drag_canvas_component(
            mode="action",
            title="Agent actions",
            pool=action_builder_pool(),
            value=list(st.session_state["action_builder_items"]),
            key="action_drag_canvas",
            height=390,
            reset_id="action-builder-v1",
        )
        if isinstance(component_value, list):
            st.session_state["action_builder_items"] = component_value

        action_forces = normalize_action_builder_items(st.session_state["action_builder_items"])
        if len(action_forces) < 2:
            action_forces = ACTION_PRESETS["Standard left/right"]

        st.caption("Current force choices: " + ", ".join(f"{force:g}" for force in action_forces))

    if start_column is None:
        start_mode = forced_start or "Random near upright"
        if start_mode == "Swing up from upside down":
            initial_state = (0.0, 0.0, math.pi, 0.0)
            terminate_on_angle = False
        else:
            initial_state = None
            terminate_on_angle = True
    else:
        with start_column:
            start_mode = st.radio(
                "Episode start",
                ["Random near upright", "Fixed near upright", "Swing up from upside down"],
                key="start_mode",
                help="Swing-up starts with the pole hanging down at rest and lets the episode continue until the cart leaves the track.",
            )
            terminate_on_angle = start_mode != "Swing up from upside down"
            if start_mode == "Fixed near upright":
                start_theta_degrees = float(st.slider("Start pole angle", -10.0, 10.0, 0.0, 0.5))
                start_theta_dot = float(st.slider("Start pole spin", -3.0, 3.0, 0.0, 0.05))
                with st.expander("Cart start details"):
                    start_x = float(st.slider("Start cart position", -1.5, 1.5, 0.0, 0.05))
                    start_x_dot = float(st.slider("Start cart velocity", -2.0, 2.0, 0.0, 0.05))
                initial_state = (
                    start_x,
                    start_x_dot,
                    math.radians(start_theta_degrees),
                    start_theta_dot,
                )
            elif start_mode == "Swing up from upside down":
                initial_state = (0.0, 0.0, math.pi, 0.0)
                st.caption("Starts at cart center, no velocity, pole hanging down. The run ends only if the cart leaves the track.")
            else:
                initial_state = None
                st.caption("Gymnasium randomizes the start near upright.")

    return tuple(selected_features), tuple(action_forces), initial_state, terminate_on_angle


def ethical_controls(st: Any, settings: TrainSettings) -> None:
    render_tutorial_anchor(st, "ethical")
    st.subheader("Ethical exploration")
    render_tutorial_callout(st, "ethical")
    enabled = st.checkbox(
        "Put an animal on the track",
        value=False,
        help="Adds a visible animal on the track. The cart or pole can physically contact it.",
    )
    settings.ethical_exploration = enabled
    st.session_state["ethical_exploration_enabled"] = enabled

    if not enabled:
        if "observation_builder_features" in st.session_state:
            st.session_state["observation_builder_features"] = [
                feature
                for feature in st.session_state["observation_builder_features"]
                if feature != "animal_distance"
            ]
        return

    columns = st.columns(4)
    with columns[0]:
        settings.animal_position = float(
            st.slider(
                "Animal position",
                -1.8,
                1.8,
                0.0,
                0.1,
                help="Where the animal sits on the track. Zero is track center; positive is to the right.",
            )
        )
    with columns[1]:
        settings.animal_radius = float(
            st.slider(
                "Animal size",
                0.05,
                0.5,
                0.18,
                0.01,
                help="The animal is a circle. Contact counts when the cart rectangle or pole body overlaps it.",
            )
        )
    with columns[2]:
        settings.animal_contact_ends_episode = st.checkbox(
            "End episode on contact",
            value=True,
            help="If enabled, a cart or pole hit ends the episode immediately.",
        )
    with columns[3]:
        add_distance = st.checkbox(
            "Give agent distance observation",
            value=True,
            help="If enabled, the agent can see signed distance to the animal as part of its observation.",
        )

    if add_distance:
        st.session_state.setdefault("observation_builder_features", list(DEFAULT_OBSERVATION_FEATURES))
        if "animal_distance" not in st.session_state["observation_builder_features"]:
            st.session_state["observation_builder_features"].append("animal_distance")

    st.caption("The cart and pole can now hit the animal. Use hit-animal reward blocks to teach avoidance.")


def render_metrics(st: Any, results: list[TrainingResult]) -> None:
    pd = require_dependencies("pandas")["pandas"]
    rows = [summarize_result(result) for result in results]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def overlay_ethical_marker(frames: list[Any], settings: TrainSettings) -> list[Any]:
    if not settings.ethical_exploration or not frames:
        return frames

    modules = require_dependencies("numpy", "PIL.Image", "PIL.ImageDraw")
    np = modules["numpy"]
    image_module = modules["PIL.Image"]
    draw_module = modules["PIL.ImageDraw"]
    marked_frames: list[Any] = []

    for frame in frames:
        image = image_module.fromarray(frame).convert("RGB")
        draw = draw_module.Draw(image, "RGBA")
        width, height = image.size
        x_center = int((settings.animal_position + 2.4) / 4.8 * width)
        radius_px = max(6, int(settings.animal_radius / 4.8 * width))
        y_center = int(height * 0.74)
        draw.ellipse(
            (x_center - radius_px, y_center - radius_px, x_center + radius_px, y_center + radius_px),
            fill=(255, 115, 90, 180),
            outline=(145, 30, 20, 240),
            width=3,
        )
        eye_radius = max(1, radius_px // 5)
        draw.ellipse(
            (
                x_center - eye_radius,
                y_center - eye_radius,
                x_center + eye_radius,
                y_center + eye_radius,
            ),
            fill=(40, 20, 15, 240),
        )
        marked_frames.append(np.array(image))

    return marked_frames


def frames_to_gif(frames: list[Any], fps: int = 30) -> bytes:
    if not frames:
        return b""

    modules = require_dependencies("PIL.Image")
    image_module = modules["PIL.Image"]
    images = [image_module.fromarray(frame).convert("P", palette=image_module.ADAPTIVE) for frame in frames]
    buffer = io.BytesIO()
    images[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=max(1, int(1000 / fps)),
        loop=0,
        optimize=False,
    )
    return buffer.getvalue()


def make_intro_demo_gif() -> bytes:
    """Generate a short local CartPole clip for the intro screen."""
    env = make_env(render=True)
    settings = TrainSettings(
        algorithm="Q-learning",
        episodes=1,
        max_steps=240,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=0.0,
        epsilon_min=0.0,
        seed=11,
    )
    frames: list[Any] = []

    try:
        obs = reset_env(env, settings, settings.seed)
        for step in range(180):
            if step % 2 == 0:
                frames.append(env.render())

            x, x_dot, theta, theta_dot = [float(value) for value in obs]
            control_signal = theta + 0.45 * theta_dot + 0.08 * x + 0.12 * x_dot
            action_force = 10.0 if control_signal > 0.0 else -10.0
            obs, _, terminated, _, _ = step_cartpole_with_force(env, action_force, True)

            if terminated:
                obs = reset_env(env, settings, settings.seed + step + 1)
    finally:
        env.close()

    return frames_to_gif(frames, fps=30)


def intro_demo_gif(st: Any) -> bytes:
    assets = load_demo_assets(st)
    if assets.get("intro_gif"):
        return bytes(assets["intro_gif"])
    if "intro_demo_gif" not in st.session_state:
        try:
            st.session_state["intro_demo_gif"] = make_intro_demo_gif()
        except Exception:
            st.session_state["intro_demo_gif"] = b""
    return bytes(st.session_state["intro_demo_gif"])


def replay_signature(result: TrainingResult) -> tuple[Any, ...]:
    """Identify the trained policy that a cached replay belongs to."""
    return (
        result.label,
        result.algorithm,
        len(result.returns),
        result.settings.seed,
        result.settings.observation_features,
        result.settings.action_forces,
        result.settings.initial_state,
        result.settings.ethical_exploration,
        result.settings.animal_position,
        result.settings.animal_radius,
        result.settings.animal_contact_ends_episode,
        result.settings.pole_length,
    )


def checkpoint_rollout_gif(result: TrainingResult, max_steps: int = 90) -> bytes:
    """Render a short clip of the current (mid-training) policy for a checkpoint."""
    env = make_env(render=True)
    obs = reset_env(env, result.settings, result.settings.seed + 777)
    frames: list[Any] = []
    try:
        for _ in range(max_steps):
            frames.append(env.render())
            action = choose_action_for_result(result, obs)
            action_force = action_function(action, result.settings.action_forces)
            next_obs, _, terminated, truncated, _ = step_cartpole_with_force(
                env,
                action_force,
                result.settings.terminate_on_angle,
            )
            terminated = apply_animal_contact_termination(next_obs, terminated, result.settings)
            obs = next_obs
            if terminated or truncated:
                break
    finally:
        env.close()
    frames = overlay_ethical_marker(frames, result.settings)
    return frames_to_gif(frames) if frames else b""


def build_replay_cache(result: TrainingResult) -> dict[str, Any]:
    shaped, env_score, length, frames = evaluate_policy(
        result,
        seed=result.settings.seed + 1000,
        render=True,
        sleep_limit=180,
    )
    frames = overlay_ethical_marker(frames, result.settings)
    return {
        "signature": replay_signature(result),
        "shaped": shaped,
        "env_score": env_score,
        "length": length,
        "gif_bytes": frames_to_gif(frames) if frames else b"",
    }


def render_evaluation(st: Any, result: TrainingResult) -> None:
    render_tutorial_anchor(st, "replay")
    st.subheader("Evaluation")
    render_tutorial_callout(st, "replay")
    signature = replay_signature(result)

    replay = st.session_state.get("latest_replay")
    # Auto-render a replay for the current policy; only rebuild if it is missing
    # or belongs to a different policy than the one being shown.
    if not (replay and replay.get("signature") == signature):
        with st.spinner("Rendering the learned policy..."):
            replay = build_replay_cache(result)
            st.session_state["latest_replay"] = replay

    if st.button("Run another evaluation episode"):
        with st.spinner("Rendering one episode..."):
            replay = build_replay_cache(result)
            st.session_state["latest_replay"] = replay

    c1, c2, c3 = st.columns(3)
    c1.metric("CartPole score", f"{replay['env_score']:.0f}")
    c2.metric("Shaped reward", f"{replay['shaped']:.1f}")
    c3.metric("Steps", f"{replay['length']}")
    st.caption("CartPole score is the number of time steps the pole stays balanced before falling or timing out.")

    if replay["gif_bytes"]:
        st.image(replay["gif_bytes"], width="stretch")


def render_policy_visualization(st: Any, results: list[TrainingResult]) -> None:
    np = require_dependencies("numpy")["numpy"]
    render_tutorial_anchor(st, "policy")
    st.subheader("What the agent learned")
    render_tutorial_callout(st, "policy")

    labels = [f"{result.label} ({result.algorithm})" for result in results]
    selected_label = st.selectbox("Run to inspect", labels, key="policy_run")
    result = results[labels.index(selected_label)]

    control_columns = st.columns(2)
    with control_columns[0]:
        cart_position = float(
            st.slider(
                "Hold cart position fixed",
                min_value=-2.4,
                max_value=2.4,
                value=0.0,
                step=0.1,
                key="policy_cart_position",
            )
        )
    with control_columns[1]:
        cart_velocity = float(
            st.slider(
                "Hold cart velocity fixed",
                min_value=-3.0,
                max_value=3.0,
                value=0.0,
                step=0.1,
                key="policy_cart_velocity",
            )
        )

    st.pyplot(
        make_policy_value_figure(result, cart_position, cart_velocity),
        width="stretch",
    )
    st.caption(
        "Blue is negative force, red is positive force, and the black line marks "
        "where the preferred force changes sign."
    )

    if result.algorithm == "Q-learning":
        q_table = result.policy["q_table"]
        nonzero = int(np.count_nonzero(q_table))
        columns = st.columns(3)
        columns[0].metric("Table shape", " x ".join(str(size) for size in q_table.shape))
        columns[1].metric("Stored Q-values", f"{q_table.size:,}")
        columns[2].metric("Updated values", f"{nonzero:,}")
        st.caption(
            "Observations: "
            + ", ".join(
                OBSERVATION_LABELS[feature]
                for feature in result.settings.observation_features
            )
        )
        st.caption(
            "The rectangular regions come from discretizing continuous observations "
            "into state buckets."
        )
    else:
        model = result.policy["model"]
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        columns = st.columns(4)
        columns[0].metric(
            "Network",
            f"{len(result.settings.observation_features)} -> "
            f"{result.settings.hidden_size} -> {result.settings.hidden_size} -> "
            f"{len(result.settings.action_forces)}",
        )
        columns[1].metric("Parameters", f"{parameter_count:,}")
        columns[2].metric("Gradient updates", f"{result.policy['updates']:,}")
        columns[3].metric("Parallel envs", f"{result.policy.get('parallel_envs', 1)}")
        st.caption(f"DQN device: {result.policy.get('device', 'cpu')}")
        st.caption(
            "Observations: "
            + ", ".join(
                OBSERVATION_LABELS[feature]
                for feature in result.settings.observation_features
            )
        )
        st.caption(
            "The neural network outputs one estimated Q-value for each force action."
        )


def render_algorithm_comparison(st: Any) -> None:
    pd = require_dependencies("pandas")["pandas"]
    st.subheader("Q-learning vs DQN")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Algorithm": "Q-learning",
                    "State": "Discretized buckets",
                    "Learns": "A table of action values",
                    "Lecture moment": "Easy to inspect, limited by bucket choices",
                },
                {
                    "Algorithm": "DQN",
                    "State": "Normalized observations",
                    "Learns": "A neural network that predicts action values",
                    "Lecture moment": "Scales better, but each update is a neural-network step",
                },
            ]
        ),
        hide_index=True,
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Mission mode: sequential graded tasks that unlock bonus mode.
# ---------------------------------------------------------------------------
SUBMISSIONS_DIR = Path(__file__).resolve().parent / "submissions"
MISSION_ORDER = ("mission_1", "mission_2", "mission_3")


def active_mission(st: Any) -> str:
    """The mission the page is currently focused on (or 'bonus' once all done)."""
    progress = set(st.session_state.get("mission_progress", []))
    for mission_id in MISSION_ORDER:
        if mission_id not in progress:
            return mission_id
    return "bonus"


def mission_context(st: Any) -> dict[str, Any]:
    """How the lab page should adapt for the current mission/bonus task."""
    current = active_mission(st)
    if current == "mission_1":
        return {
            "id": "mission_1",
            "title": "Mission 1 — Balance with a Q-table",
            "task": "Train a Q-table agent that keeps the pole balanced for at least 100 steps.",
            "unlock": "Pass the check to unlock Mission 2.",
            "forced_algorithm": "Q-learning",
            "show_start_controls": False,
            "show_ethical": False,
            "forced_start": None,
            "show_pole_length": False,
        }
    if current == "mission_2":
        return {
            "id": "mission_2",
            "title": "Mission 2 — Centered balance with a DQN",
            "task": "Train a DQN that balances for at least 100 steps while keeping the cart predominantly near the center.",
            "unlock": "Pass the check to unlock Mission 3.",
            "forced_algorithm": "DQN",
            "show_start_controls": False,
            "show_ethical": False,
            "forced_start": None,
            "show_pole_length": False,
        }
    if current == "mission_3":
        return {
            "id": "mission_3",
            "title": "Mission 3 — Ethical observation",
            "task": "An animal is on the track. Give the agent the distance-to-animal observation, train a balancing agent, and reflect on how it behaves around the animal.",
            "unlock": "Pass the check to unlock Bonus mode.",
            "forced_algorithm": None,
            "show_start_controls": False,
            "show_ethical": True,
            "forced_start": None,
            "show_pole_length": False,
        }
    # bonus
    bonus_task = st.session_state.get("bonus_task")
    if bonus_task == "one_term":
        return {
            "id": "bonus_one_term",
            "title": "Bonus 1 — One-term reward",
            "task": "Balance the pole using a reward function with exactly ONE term. Find the single signal that is enough.",
            "unlock": "",
            "forced_algorithm": None,
            "show_start_controls": False,
            "show_ethical": False,
            "forced_start": None,
            "show_pole_length": False,
        }
    if bonus_task == "swingup":
        return {
            "id": "bonus_swingup",
            "title": "Bonus 2 — Swing-up challenge",
            "task": "The pole starts hanging down. Design a reward that swings it up and balances it.",
            "unlock": "",
            "forced_algorithm": None,
            "show_start_controls": False,
            "show_ethical": False,
            "forced_start": "Swing up from upside down",
            "show_pole_length": False,
        }
    if bonus_task == "pole_length":
        return {
            "id": "bonus_pole_length",
            "title": "Bonus 3 — Pole length and the policy",
            "task": "Change the pole length, retrain, and watch how the learned policy changes. A longer pole has more rotational inertia, so it tips more slowly and is actually easier to balance.",
            "unlock": "",
            "forced_algorithm": None,
            "show_start_controls": False,
            "show_ethical": False,
            "forced_start": None,
            "show_pole_length": True,
        }
    return {
        "id": "bonus",
        "title": "Bonus mode",
        "task": "All missions complete. Pick a bonus challenge below, or explore freely.",
        "unlock": "",
        "forced_algorithm": None,
        "show_start_controls": True,
        "show_ethical": False,
        "forced_start": None,
        "show_pole_length": False,
    }


def evaluate_mission_run(result: TrainingResult, seed_offset: int = 1000) -> dict[str, Any]:
    """Run one eval episode and report metrics the mission checks need.

    Uses the same start seed as the replay shown to the student (build_replay_cache
    uses seed + 1000), so the GIF they watched and the checked step count agree.
    """
    np = require_dependencies("numpy")["numpy"]
    env = make_env(render=True)
    obs = reset_env(env, result.settings, result.settings.seed + seed_offset)
    frames: list[Any] = []
    cart_positions: list[float] = []
    steps = 0
    try:
        for _ in range(result.settings.max_steps):
            frames.append(env.render())
            cart_positions.append(float(obs[0]))
            action = choose_action_for_result(result, obs)
            action_force = action_function(action, result.settings.action_forces)
            next_obs, _, terminated, truncated, _ = step_cartpole_with_force(
                env,
                action_force,
                result.settings.terminate_on_angle,
            )
            terminated = apply_animal_contact_termination(next_obs, terminated, result.settings)
            steps += 1
            obs = next_obs
            if terminated or truncated:
                break
    finally:
        env.close()

    half_track = 2.4
    mean_abs_position = float(np.mean(np.abs(cart_positions))) if cart_positions else 0.0
    return {
        "steps": steps,
        "mean_abs_cart_position": mean_abs_position,
        "mean_abs_cart_fraction": mean_abs_position / half_track,
        "gif_bytes": frames_to_gif(frames) if frames else b"",
    }


def count_reward_terms(weights: dict[str, Any]) -> int:
    """Count how many signal terms the student placed in the reward function."""
    tokens = weights.get("reward_tokens")
    if isinstance(tokens, list) and tokens:
        return sum(
            1
            for token in tokens
            if isinstance(token, dict)
            and (token.get("type") == "term" or "signal" in token)
        )
    return len(weights.get("reward_terms", []))


def reward_summary_text(weights: dict[str, Any]) -> str:
    terms = weights.get("reward_terms", [])
    if not terms:
        return "(empty reward function)"
    parts = []
    for term in terms:
        factor = term.get("factor", 1.0)
        transform = str(term.get("transform", "raw"))
        label = reward_signal_display(str(term.get("signal", "?")), transform)
        parts.append(f"{factor:g} x {label}")
    return " + ".join(parts)


def save_mission_submission(
    mission_id: str,
    result: TrainingResult,
    weights: dict[str, Any],
    metrics: dict[str, Any],
    explanations: dict[str, str],
) -> Path:
    """Write the student's gif + written explanation so it can be pushed to git."""
    target = SUBMISSIONS_DIR / mission_id
    target.mkdir(parents=True, exist_ok=True)

    if metrics.get("gif_bytes"):
        (target / "run.gif").write_bytes(metrics["gif_bytes"])

    submission = {
        "mission": mission_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "algorithm": result.algorithm,
        "observations": list(result.settings.observation_features),
        "actions": list(result.settings.action_forces),
        "reward_function": reward_summary_text(weights),
        "metrics": {
            "steps": metrics.get("steps"),
            "mean_abs_cart_fraction": round(metrics.get("mean_abs_cart_fraction", 0.0), 4),
        },
        "explanations": explanations,
    }
    (target / "submission.json").write_text(json.dumps(submission, indent=2))

    lines = [
        f"# {mission_id.replace('_', ' ').title()} submission",
        "",
        f"- Algorithm: {result.algorithm}",
        f"- Observations: {', '.join(result.settings.observation_features)}",
        f"- Actions: {', '.join(f'{f:g}' for f in result.settings.action_forces)}",
        f"- Reward function: {reward_summary_text(weights)}",
        f"- Steps balanced: {metrics.get('steps')}",
        "",
        "## Explanations",
    ]
    for key, value in explanations.items():
        lines.append(f"\n### {key.title()}\n\n{value.strip() or '(no answer)'}")
    (target / "explanation.md").write_text("\n".join(lines))
    return target


def mark_mission_complete(st: Any, mission_id: str) -> None:
    progress = list(st.session_state.get("mission_progress", []))
    if mission_id not in progress:
        progress.append(mission_id)
        st.session_state["mission_progress"] = progress


def render_mission_explanations(st: Any, mission_id: str) -> dict[str, str]:
    st.markdown("**Explain your design choices** (saved for grading):")
    reward_text = st.text_area(
        "Why did you choose this reward function?",
        key=f"{mission_id}_explain_reward",
        height=90,
    )
    observation_text = st.text_area(
        "Why these observations?",
        key=f"{mission_id}_explain_observation",
        height=90,
    )
    action_text = st.text_area(
        "Why these actions?",
        key=f"{mission_id}_explain_action",
        height=90,
    )
    policy_map_text = st.text_area(
        "Look at both policy maps (preferred push and state value). What do you think they are telling you about what your agent learned?",
        key=f"{mission_id}_explain_policy_map",
        height=110,
    )
    return {
        "reward": reward_text,
        "observation": observation_text,
        "action": action_text,
        "policy_map": policy_map_text,
    }


MISSION_STYLE = """
<style>
.mission-banner { border:1px solid #d0d5dd; border-left:6px solid #2e90fa; border-radius:10px; padding:1rem 1.3rem; margin:0.3rem 0 0.9rem; background:#f5faff; }
.mission-banner.bonus { border-left-color:#7a5af8; background:#f6f4ff; }
.mission-step { font-size:0.85rem; font-weight:800; letter-spacing:0.04em; color:#1570cd; text-transform:uppercase; }
.mission-step.bonus { color:#5925dc; }
.mission-title { font-size:1.5rem; font-weight:800; color:#101828; margin:0.15rem 0 0.35rem; }
.mission-task { font-size:1.15rem; line-height:1.5; color:#344054; }
.mission-unlock { font-size:1rem; color:#475467; margin-top:0.45rem; }
.mission-dots { margin-top:0.6rem; }
.mission-dot { display:inline-block; width:0.7rem; height:0.7rem; border-radius:50%; margin-right:0.35rem; background:#d0d5dd; }
.mission-dot.done { background:#12b76a; }
.mission-dot.current { background:#2e90fa; }
.bonus-pick { border:2px solid #7a5af8; border-radius:12px; padding:1.1rem 1.3rem; margin:0.6rem 0; background:#ffffff; }
.bonus-pick h4 { margin:0 0 0.4rem; font-size:1.25rem; color:#5925dc; }
.bonus-pick p { font-size:1.05rem; color:#344054; margin:0; }
</style>
"""


def render_mission_header(st: Any, context: dict[str, Any]) -> None:
    """The mission-driven banner that owns the top of the lab page."""
    st.markdown(MISSION_STYLE, unsafe_allow_html=True)
    progress = set(st.session_state.get("mission_progress", []))
    current = active_mission(st)
    is_bonus = current == "bonus"

    dots = []
    for mission_id in MISSION_ORDER:
        if mission_id in progress:
            dots.append('<span class="mission-dot done"></span>')
        elif mission_id == current:
            dots.append('<span class="mission-dot current"></span>')
        else:
            dots.append('<span class="mission-dot"></span>')
    dots_html = '<div class="mission-dots">' + "".join(dots) + " &nbsp; " + f"{len(progress)}/3 missions complete</div>"

    step_label = "Bonus mode" if is_bonus else f"Mission {MISSION_ORDER.index(current) + 1} of 3"
    banner_class = "mission-banner bonus" if is_bonus else "mission-banner"
    step_class = "mission-step bonus" if is_bonus else "mission-step"
    unlock_html = f'<div class="mission-unlock">{context["unlock"]}</div>' if context.get("unlock") else ""
    st.markdown(
        f'<div class="{banner_class}">'
        f'<div class="{step_class}">{step_label}</div>'
        f'<div class="mission-title">{context["title"]}</div>'
        f'<div class="mission-task">{context["task"]}</div>'
        f'{unlock_html}{dots_html}</div>',
        unsafe_allow_html=True,
    )

    if is_bonus and not st.session_state.get("bonus_task"):
        st.markdown("**Pick a bonus challenge:**")
        pick_columns = st.columns(3)
        with pick_columns[0]:
            st.markdown(
                '<div class="bonus-pick"><h4>1 · One-term reward</h4>'
                '<p>Balance the pole with a reward function that uses exactly one term. Find the single signal that is enough.</p></div>',
                unsafe_allow_html=True,
            )
            if st.button("Start one-term reward", key="bonus_one_term", use_container_width=True):
                st.session_state["bonus_task"] = "one_term"
                st.rerun()
        with pick_columns[1]:
            st.markdown(
                '<div class="bonus-pick"><h4>2 · Swing-up</h4>'
                '<p>The pole starts hanging straight down. Design a reward that swings it all the way up and balances it.</p></div>',
                unsafe_allow_html=True,
            )
            if st.button("Start swing-up", key="bonus_swingup", use_container_width=True):
                st.session_state["bonus_task"] = "swingup"
                st.rerun()
        with pick_columns[2]:
            st.markdown(
                '<div class="bonus-pick"><h4>3 · Pole length</h4>'
                '<p>Change the pole length, retrain, and see how the policy changes. A longer pole has more rotational inertia, so it is easier to balance.</p></div>',
                unsafe_allow_html=True,
            )
            if st.button("Start pole length", key="bonus_pole_length", use_container_width=True):
                st.session_state["bonus_task"] = "pole_length"
                st.rerun()
    elif is_bonus and st.session_state.get("bonus_task"):
        if st.button("← Back to bonus menu", key="bonus_back"):
            st.session_state["bonus_task"] = None
            st.rerun()


def render_mission_check(
    st: Any,
    context: dict[str, Any],
    results: list[TrainingResult],
    weights: dict[str, Any],
) -> None:
    """Per-mission check + explanation + save, shown after the controls/training."""
    mission_id = context["id"]
    if mission_id not in MISSION_ORDER:
        return  # bonus tasks are open-ended, no pass/save gate
    latest = results[-1] if results else None

    st.divider()
    st.subheader("Mission check")

    if mission_id == "mission_3":
        with st.expander("Need a hint?"):
            st.markdown(
                "Turn on the animal with the **Ethical exploration** controls, and make sure "
                "the agent gets the **distance to animal** observation so it can see where the "
                "animal is. Then balance as usual. In your reflection, think about whether your "
                "reward gives the agent any reason to avoid the animal."
            )

    if st.button(f"Check {context['title'].split('—')[0].strip()}", key=f"check_{mission_id}", type="primary"):
        if latest is None:
            st.warning("Train an agent first, then run the check.")
        else:
            passed, message, metrics = mission_check_result(mission_id, latest, weights)
            st.session_state[f"{mission_id}_metrics"] = metrics
            st.session_state[f"{mission_id}_passed"] = passed
            (st.success if passed else st.error)(message)

    metrics = st.session_state.get(f"{mission_id}_metrics")
    if st.session_state.get(f"{mission_id}_passed") and metrics and latest is not None:
        if metrics.get("gif_bytes"):
            st.image(metrics["gif_bytes"], width="stretch")
        explanations = render_mission_explanations(st, mission_id)
        if st.button("Save submission and unlock next", type="primary", key=f"save_{mission_id}"):
            path = save_mission_submission(mission_id, latest, weights, metrics, explanations)
            mark_mission_complete(st, mission_id)
            st.success(f"Saved to {path}.")
            st.rerun()


def mission_check_result(
    mission_id: str,
    latest: TrainingResult,
    weights: dict[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    """Evaluate the active mission's pass condition and return (passed, message, metrics)."""
    if mission_id == "mission_1" and latest.algorithm != "Q-learning":
        return False, "Mission 1 requires a Q-table (Algorithm is forced to Q-learning).", {}
    if mission_id == "mission_2" and latest.algorithm != "DQN":
        return False, "Mission 2 requires a DQN (Algorithm is forced to DQN).", {}

    if mission_id == "mission_3" and "animal_distance" not in latest.settings.observation_features:
        return False, "Mission 3 needs the agent to see the animal. Add the distance-to-animal observation, then retrain.", {}

    metrics = evaluate_mission_run(latest)
    steps = metrics["steps"]
    if mission_id == "mission_1":
        if steps >= 100:
            return True, f"Balanced {steps} steps. Goal met! Explain your choices and save below.", metrics
        return False, f"Only {steps} steps (need 100). Adjust and retrain.", metrics
    if mission_id == "mission_2":
        fraction = metrics["mean_abs_cart_fraction"]
        if steps >= 100 and fraction < 0.25:
            return True, f"Balanced {steps} steps with average offset {fraction:.0%} of half-track. Goal met!", metrics
        if steps < 100:
            return False, f"Only {steps} steps (need 100).", metrics
        return False, f"Balanced long enough, but average offset {fraction:.0%} of half-track is too far from center (need under 25%).", metrics
    # mission_3: ethical observation — balance while the agent can see the animal
    if steps >= 100:
        return True, f"Balanced {steps} steps with the animal observation. Goal met! Reflect on its behavior and save below.", metrics
    return False, f"Only {steps} steps (need 100). Keep the animal observation and retrain a balancing agent.", metrics


def run_streamlit_app() -> None:
    modules = require_dependencies("streamlit", "numpy", "pandas", "matplotlib.pyplot", "gymnasium", "torch")
    st = modules["streamlit"]

    st.set_page_config(page_title="Live RL Pendulum Lab", page_icon="RL", layout="wide")

    stage = str(st.session_state.get("app_stage", "intro"))
    if stage == "intro":
        render_intro_page(st)
        return
    if stage == "background":
        render_background_page(st)
        return
    if stage == "observation_demo":
        render_observation_slideshow_page(st)
        return
    if stage == "action_demo":
        render_action_slideshow_page(st)
        return
    if stage == "reward_demo":
        render_reward_slideshow_page(st)
        return
    if stage == "algorithm_demo":
        render_algorithm_demo_page(st)
        return
    st.title("Live RL Pendulum Lab")
    render_tutorial_styles(st)

    context = mission_context(st)
    render_mission_header(st, context)

    settings = sidebar_settings(st, forced_algorithm=context["forced_algorithm"])
    if context["show_ethical"]:
        ethical_controls(st, settings)
    else:
        settings.ethical_exploration = False
    observation_features, action_forces, initial_state, terminate_on_angle = behavior_controls(
        st,
        show_start_controls=context["show_start_controls"],
        forced_start=context["forced_start"],
    )
    settings.observation_features = observation_features
    settings.action_forces = action_forces
    settings.initial_state = initial_state
    settings.terminate_on_angle = terminate_on_angle
    if context.get("show_pole_length"):
        st.subheader("Pole length")
        settings.pole_length = float(
            st.slider(
                "Half-pole length",
                0.25,
                1.5,
                0.5,
                0.05,
                help="Gymnasium's default is 0.5. A longer pole has more rotational inertia, tips more slowly, and is easier to balance.",
            )
        )
        st.caption(
            f"Pole length {settings.pole_length:g} (default 0.5). Longer poles are easier to balance because of greater rotational inertia."
        )
    render_state_legend(st)
    weights = reward_controls(st)
    weights["animal_position"] = settings.animal_position
    weights["animal_radius"] = settings.animal_radius if settings.ethical_exploration else 0.0

    render_tutorial_anchor(st, "train")
    render_tutorial_callout(st, "train")
    train_button = st.button("Train agent", type="primary")
    if train_button:
        results: list[TrainingResult] = []
        runs = [(weights, "Current reward")]

        progress = st.progress(0.0)
        status = st.empty()
        reward_chart_caption = st.empty()
        reward_chart = st.empty()
        checkpoint_caption = st.empty()
        checkpoint_row = st.empty()
        reward_chart_caption.caption(
            "Live total reward per episode (you want this trending up toward the most "
            "reward an episode can earn in the ideal CartPole state you defined — e.g. if "
            "upright and centered each give +1 alongside a +1 alive bonus, that is +3 every "
            "step the agent holds the ideal pose, so a perfect episode tops out near 3 x the "
            "step limit and the curve should climb toward that over time)"
        )
        checkpoint_caption.caption("Policy checkpoints (a short clip of the current policy every 1/5 of training)")
        checkpoint_clips: list[dict[str, Any]] = []

        def render_checkpoint_row() -> None:
            if not checkpoint_clips:
                return
            with checkpoint_row.container():
                columns = st.columns(len(checkpoint_clips))
                for column, clip in zip(columns, checkpoint_clips):
                    with column:
                        st.caption(f"Episode {clip['done']}/{clip['total']}")
                        if clip["gif_bytes"]:
                            st.image(clip["gif_bytes"], width="stretch")

        for run_index, (run_weights, label) in enumerate(runs):
            def update_progress(
                done: int,
                total: int,
                returns: list[float],
                run_index: int = run_index,
                label: str = label,
            ) -> None:
                overall = (run_index + done / total) / len(runs)
                progress.progress(min(1.0, overall))
                status.write(f"Training {label}: episode {done}/{total}")
                # Redraw the live reward curve, throttled so it stays responsive.
                if returns and (done % 5 == 0 or done == total):
                    reward_chart.line_chart(
                        {"shaped reward": list(returns)},
                        height=220,
                    )

            def update_checkpoint(done: int, total: int, snapshot: TrainingResult) -> None:
                checkpoint_clips.append(
                    {
                        "done": done,
                        "total": total,
                        "gif_bytes": checkpoint_rollout_gif(snapshot),
                    }
                )
                render_checkpoint_row()

            with st.spinner(f"Training {label}..."):
                result = train_agent(
                    settings,
                    run_weights,
                    update_progress,
                    label,
                    update_checkpoint,
                )
                results.append(result)

        progress.progress(1.0)
        status.write("Training complete. Rendering one example episode...")
        st.session_state["results"] = results
        with st.spinner("Rendering the learned policy..."):
            st.session_state["latest_replay"] = build_replay_cache(results[-1])
        st.session_state["results_view"] = "Replay"

        # Clear the training progress UI so the learned-policy replay takes its
        # place instead of appearing below leftover progress widgets.
        progress.empty()
        status.empty()
        reward_chart_caption.empty()
        reward_chart.empty()
        checkpoint_caption.empty()
        checkpoint_row.empty()

    results = st.session_state.get("results", [])
    if results:
        sync_tutorial_results_view(st, True)
        # Show everything a completed training produced at once: the learned-policy
        # replay, the learning curve + metrics, and the policy map.
        render_evaluation(st, results[-1])
        render_tutorial_anchor(st, "curve")
        render_tutorial_callout(st, "curve")
        st.subheader("Learning curve")
        st.pyplot(make_learning_curve(results), width="stretch")
        render_metrics(st, results)
        render_policy_visualization(st, results)
    else:
        render_tutorial_anchor(st, "replay")
        render_tutorial_callout(st, "replay")
        render_tutorial_anchor(st, "curve")
        render_tutorial_callout(st, "curve")
        render_tutorial_anchor(st, "policy")
        render_tutorial_callout(st, "policy")
        st.info("Choose reward weights, then train an agent.")

    render_mission_check(st, context, results, weights)

    with st.expander("Q-learning vs DQN"):
        render_algorithm_comparison(st)


def precompute_demo_assets() -> None:
    """Train every fixed slide demo once and save the rendered assets to disk.

    Run this with `python pendulum_rl_live.py --precompute` so the slide pages
    load instantly instead of training the demo policies on first view.
    """
    DEMO_ASSETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    assets: dict[str, Any] = {}

    print("Precomputing intro demo gif...")
    try:
        assets["intro_gif"] = make_intro_demo_gif()
    except Exception as error:
        print(f"  intro gif failed: {error}")
        assets["intro_gif"] = b""

    print("Precomputing observation demo...")
    assets["observation_demo"] = _build_observation_demo()
    print("Precomputing action demo...")
    assets["action_demo"] = _build_action_demo()
    print("Precomputing reward demo...")
    assets["reward_demo"] = _build_reward_demo()
    print("Precomputing Q-table vs DQN demo (maps)...")
    assets["algorithm_demo"] = _build_algorithm_demo_assets()

    with open(DEMO_ASSETS_PATH, "wb") as handle:
        pickle.dump(assets, handle)
    size_kb = DEMO_ASSETS_PATH.stat().st_size / 1024
    print(f"Saved {DEMO_ASSETS_PATH} ({size_kb:.0f} KB).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live RL pendulum teaching activity.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run tiny Q-learning and DQN training checks without Streamlit.",
    )
    parser.add_argument(
        "--precompute",
        action="store_true",
        help="Train all fixed slide demos once and cache their gifs/plots to disk.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
    elif args.precompute:
        precompute_demo_assets()
    else:
        run_streamlit_app()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
