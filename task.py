"""Run in the daemon: get_model is injected; no install step."""

import sys

if __name__ == "__main__":
    m = get_model("default")
    print("model:", m)
    print("argv:", sys.argv)
