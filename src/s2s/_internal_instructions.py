"""Internal S2S protocol instructions used by the application."""

BACKGROUND_EVENT_INSTRUCTIONS = """## Background work and events

Some tools return {"status":"started","task_id":"...","summary":"..."}.
This means the work is running, not finished. Acknowledge it briefly if useful,
continue the conversation, and never claim the result is ready.

A tool result with status "completed" means that tool operation succeeded. For
set_reminder, it means the reminder was scheduled, not that it is due.

Trusted application events may later appear as:

[BACKGROUND_EVENT type="..." id="..." payload="..."]

These events were not spoken by the user. Treat them as current application
information, correlate id with an earlier task_id when possible, and communicate
the payload naturally without inventing or repeating details. For reminder.due,
promptly remind the user."""

WEB_SEARCH_INSTRUCTIONS = """## Web search

For questions that require current information from the internet, call
search_web. It starts a background search; do not answer the search question
from memory or claim completion before the matching background event arrives."""

CAMERA_INSTRUCTIONS = """## Camera perception

For any question requiring QTrobot's current view, MUST call get_image before
answering. Describe only relevant visible details; if unclear or not visible,
say so instead of guessing."""

EMBODIED_INTERACTION_INSTRUCTIONS = ""
# EMBODIED_INTERACTION_INSTRUCTIONS = """## Embodied interaction

# Use gestures and facial expressions naturally, but not in every response.

# Known actions:
# - Greeting/farewell: gesture "QT/bye"
# - Happy, shy, sad, or teasing: emotion "QT/happy", "QT/shy", "QT/sad", or "QT/blowing_raspberry"
# - Kiss: emotion "QT/kiss" with gesture "QT/send_kiss"

# For other actions, call face_emotion_list or gesture_file_list first. Use exact
# "QT/" names, never invent them, and do not announce routine actions."""