"""Shared utility functions across the project."""

def get_system_prompt(task_type):
    """Returns the system prompt based on the specific task."""
    if task_type == "english":
        return "You are a helpful assistant. Summarize the following text in English."
    elif task_type == "french":
        return "You are a helpful assistant. Summarize the following text in French."
    elif task_type == "crosslingual":
        return "You are a helpful assistant. Summarize the following French text in English."
    else:
        raise ValueError(f"Unknown task type: {task_type}")

def get_bertscore_language(task_type):
    """Returns the appropriate language for BERTScore based on task type."""
    if task_type == "english" or task_type == "crosslingual":
        return "en"
    elif task_type == "french":
        return "fr"
    else:
        raise ValueError(f"Unknown task type: {task_type}")
