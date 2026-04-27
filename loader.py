"""Optional models present before any task runs. Lazy HF loads still go through get_model(id)."""


def load_models():
    return {}
