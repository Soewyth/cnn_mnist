import datetime


def get_run_id(tag: str = None) -> str:
    """Generates a unique run ID based on timestamp and tag."""
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if tag:
        return f"{tag}_{run_id}"
    return run_id