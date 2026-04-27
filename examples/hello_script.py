"""Example script run via: modeldaemon run examples/hello_script.py"""

from modeldaemon.runtime import get_model, list_models

if __name__ == "__main__":
    print("models:", list_models())
    m = get_model("default")
    print("default model:", m)
