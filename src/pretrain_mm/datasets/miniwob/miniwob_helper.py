#
import gymnasium
import torch
from miniwob.action import ActionTypes

from pretrain_mm import logger


def eval(
    model: torch.nn.Module,
    obs_process_func: callable,
    env_name: str = "miniwob/click-test-2-v1",
    render_mode: str = None,
):
    env = gymnasium.make(env_name, render_mode=render_mode)

    # Wrap the code in try-finally to ensure proper cleanup.
    try:
        # Start a new episode.
        obs, info = env.reset()
        action_type, ref = obs_process_func(model, obs, info=info)
        assert action_type in ActionTypes, f"action_type from obs_process_func not correct: {action_type} | ref: {ref}"

        action = env.unwrapped.create_action(action_type, ref=ref)
        obs, reward, terminated, truncated, info = env.step(action)

        logger.log(f"Reward: {reward} | info: {info}")
        # Check if the action was correct.
    finally:
        env.close()

    return reward
