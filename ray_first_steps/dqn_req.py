import requests
from dqn_tune import lula_genesis
import time

if __name__ == "__main__":

    env = lula_genesis(None)
    obs, info = env.reset()
    
    while True:
        print("-> Requesting action for obs ...")
        # Send a request to serve.
        resp = requests.get("http://localhost:8000/", json={"observation": obs.tolist()},)
        response = resp.json()
        print("<- Received response {}".format(response))

        # Apply the action in the env.
        action = response["action"]
        obs, reward, done, _, _ = env.step(action)

        # If episode done -> reset to get initial observation of new episode.
        if done:
            print("LANDED!")
            obs, info = env.reset()
        else:
            env.render()