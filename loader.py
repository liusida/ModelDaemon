"""Called once when the daemon starts. Return name -> object (your real model here)."""


def load_models():
    return {
        "default": {"kind": "stub", "note": "swap in torch.load(...) etc."},
    }
