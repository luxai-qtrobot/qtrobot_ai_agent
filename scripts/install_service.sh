#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="qtrobot-ai-agent.service"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
MAIN_SCRIPT="$PROJECT_DIR/src/main.py"
CONFIG_FILE="$PROJECT_DIR/config/config.yaml"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Missing virtual environment: $VENV_PYTHON" >&2
    echo "Create .venv and install requirements before installing the service." >&2
    exit 1
fi
if [[ ! -f "$MAIN_SCRIPT" || ! -f "$CONFIG_FILE" ]]; then
    echo "The project is incomplete: main.py or config.yaml is missing." >&2
    exit 1
fi

SERVICE_USER="${SUDO_USER:-$(id -un)}"
if [[ "$SERVICE_USER" == "root" ]]; then
    if id qtrobot >/dev/null 2>&1; then
        SERVICE_USER="qtrobot"
    else
        echo "Run this script as the user who should run the application." >&2
        exit 1
    fi
fi
SERVICE_GROUP="$(id -gn "$SERVICE_USER")"
SERVICE_HOME="$(getent passwd "$SERVICE_USER" | cut -d: -f6)"

systemd_escape() {
    printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/%/%%/g'
}

PROJECT_DIR_ESCAPED="$(systemd_escape "$PROJECT_DIR")"
VENV_PYTHON_ESCAPED="$(systemd_escape "$VENV_PYTHON")"
MAIN_SCRIPT_ESCAPED="$(systemd_escape "$MAIN_SCRIPT")"
CONFIG_FILE_ESCAPED="$(systemd_escape "$CONFIG_FILE")"
SERVICE_HOME_ESCAPED="$(systemd_escape "$SERVICE_HOME")"

TEMP_UNIT="$(mktemp)"
trap 'rm -f "$TEMP_UNIT"' EXIT

cat >"$TEMP_UNIT" <<EOF
[Unit]
Description=QTrobot AI Agent
After=network-online.target qtrobot-llama-cpp.service luxai-s2s-magpie.service
Wants=network-online.target qtrobot-llama-cpp.service luxai-s2s-magpie.service

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_GROUP
WorkingDirectory=$PROJECT_DIR_ESCAPED
ExecStart="$VENV_PYTHON_ESCAPED" "$MAIN_SCRIPT_ESCAPED" "$CONFIG_FILE_ESCAPED"
Environment="HOME=$SERVICE_HOME_ESCAPED"
Environment="PYTHONUNBUFFERED=1"
Restart=on-failure
RestartSec=5
KillSignal=SIGINT
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF

echo "Installing $SERVICE_NAME for $SERVICE_USER from $PROJECT_DIR"
sudo install -m 0644 "$TEMP_UNIT" "/etc/systemd/system/$SERVICE_NAME"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

echo
echo "QTrobot AI Agent is installed, enabled, and running."
echo "Check it with: sudo systemctl status $SERVICE_NAME"
echo "Follow logs with: sudo journalctl -u $SERVICE_NAME -f"
