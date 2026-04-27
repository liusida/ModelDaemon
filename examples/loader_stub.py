"""Example loader: no ML deps; replace with real torch.load / transformers, etc."""


def load_models():
    # Keys are names passed to get_model("name").
    return {
        "default": {"kind": "stub", "params": 1_000_000},
        "small": [1, 2, 3],
    }
