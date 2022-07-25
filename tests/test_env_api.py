from gym_minipupper.envs.minipupper_env import MinipupperEnv
from gym.utils.env_checker import check_env

if __name__ == "__main__":
    check_env(MinipupperEnv(render=False))
