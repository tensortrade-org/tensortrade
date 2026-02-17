#!/usr/bin/env bash
# dev.sh — Start TensorTrade backend + frontend with process management
#
# Usage:
#   ./dev.sh          Start both servers
#   ./dev.sh stop     Stop all running servers
#   ./dev.sh status   Check if servers are running
#   ./dev.sh init-db  Initialize database and show seed data counts
#   ./dev.sh reset-db Delete database and re-seed from scratch

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_DIR="$ROOT_DIR/.dev-pids"
LOG_DIR="$ROOT_DIR/.dev-logs"
ENV_FILE="$ROOT_DIR/.env"

mkdir -p "$PID_DIR" "$LOG_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

load_env() {
    if [ -f "$ENV_FILE" ]; then
        while IFS= read -r line || [ -n "$line" ]; do
            line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
            [[ -z "$line" || "$line" == \#* ]] && continue
            key="${line%%=*}"
            value="${line#*=}"
            export "$key=$value"
        done < "$ENV_FILE"
        echo -e "${GREEN}Loaded .env${NC}"
    fi
}

is_running() {
    local pid_file="$PID_DIR/$1.pid"
    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        rm -f "$pid_file"
    fi
    return 1
}

stop_process() {
    local name="$1"
    local pid_file="$PID_DIR/$name.pid"
    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${YELLOW}Stopping $name (PID $pid)...${NC}"
            kill "$pid" 2>/dev/null || true
            # Wait up to 5 seconds for graceful shutdown
            for _ in $(seq 1 10); do
                kill -0 "$pid" 2>/dev/null || break
                sleep 0.5
            done
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
            echo -e "${GREEN}Stopped $name${NC}"
        fi
        rm -f "$pid_file"
    fi
}

cmd_stop() {
    stop_process "backend"
    stop_process "frontend"
    # Kill any orphan processes on our ports
    lsof -ti:3000 2>/dev/null | xargs kill -9 2>/dev/null || true
    lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null || true
    echo -e "${GREEN}All servers stopped.${NC}"
}

cmd_status() {
    if is_running "backend"; then
        echo -e "${GREEN}Backend:  running (PID $(cat "$PID_DIR/backend.pid"))${NC}"
    else
        echo -e "${RED}Backend:  not running${NC}"
    fi
    if is_running "frontend"; then
        echo -e "${GREEN}Frontend: running (PID $(cat "$PID_DIR/frontend.pid"))${NC}"
    else
        echo -e "${RED}Frontend: not running${NC}"
    fi
}

cmd_start() {
    # Stop any existing processes first
    stop_process "backend"
    stop_process "frontend"

    load_env

    # Database is auto-initialized on backend startup (tables + seed data)
    echo -e "${CYAN}Starting backend (FastAPI :8000)...${NC}"
    "$ROOT_DIR/.venv/bin/python" -m tensortrade_platform.api.server \
        > "$LOG_DIR/backend.log" 2>&1 &
    echo $! > "$PID_DIR/backend.pid"

    # Wait for backend to be ready
    for i in $(seq 1 20); do
        if curl -sf http://localhost:8000/api/status > /dev/null 2>&1; then
            echo -e "${GREEN}Backend ready.${NC}"
            break
        fi
        if [ "$i" -eq 20 ]; then
            echo -e "${RED}Backend failed to start. Check $LOG_DIR/backend.log${NC}"
            exit 1
        fi
        sleep 0.5
    done

    # Clear stale Next.js cache to avoid MODULE_NOT_FOUND errors
    rm -rf "$ROOT_DIR/dashboard/.next"

    # Ensure node_modules exist (e.g. after branch switch)
    if [ ! -d "$ROOT_DIR/dashboard/node_modules" ]; then
        echo -e "${YELLOW}Installing frontend dependencies...${NC}"
        (cd "$ROOT_DIR/dashboard" && npm install --silent)
    fi

    echo -e "${CYAN}Starting frontend (Next.js :3000)...${NC}"
    cd "$ROOT_DIR/dashboard" && npm run dev \
        > "$LOG_DIR/frontend.log" 2>&1 &
    echo $! > "$PID_DIR/frontend.pid"
    cd "$ROOT_DIR"

    sleep 2
    echo ""
    echo -e "${GREEN}====================================${NC}"
    echo -e "${GREEN}  TensorTrade Dev Servers Running${NC}"
    echo -e "${GREEN}====================================${NC}"
    echo -e "  Backend:  ${CYAN}http://localhost:8000${NC}"
    echo -e "  Frontend: ${CYAN}http://localhost:3000${NC}"
    echo -e "  Logs:     ${YELLOW}$LOG_DIR/${NC}"
    echo ""
    echo -e "  Stop with: ${YELLOW}./dev.sh stop${NC}"
    echo -e "  Status:    ${YELLOW}./dev.sh status${NC}"
    echo -e "${GREEN}====================================${NC}"
}

cmd_reset_db() {
    local db_path="$HOME/.tensortrade/experiments.db"
    if [ -f "$db_path" ]; then
        echo -e "${YELLOW}Removing database: $db_path${NC}"
        rm -f "$db_path" "${db_path}-wal" "${db_path}-shm"
        echo -e "${GREEN}Database removed.${NC}"
    else
        echo -e "${YELLOW}No database found at $db_path${NC}"
    fi
    echo -e "${CYAN}Initializing fresh database with seed data...${NC}"
    "$ROOT_DIR/.venv/bin/python" -c "
from tensortrade_platform.training.experiment_store import ExperimentStore
from tensortrade_platform.training.hyperparameter_store import HyperparameterStore
from tensortrade_platform.training.dataset_store import DatasetStore
store = ExperimentStore()
hp = HyperparameterStore()
ds = DatasetStore()
print(f'  Datasets seeded:  {len(ds.list_datasets())}')
print(f'  HP packs seeded:  {len(hp.list_packs())}')
store.close(); hp.close(); ds.close()
"
    echo -e "${GREEN}Database initialized with seed datasets and HP packs.${NC}"
}

cmd_init_db() {
    local db_path="$HOME/.tensortrade/experiments.db"
    echo -e "${CYAN}Initializing database (seeding if empty)...${NC}"
    "$ROOT_DIR/.venv/bin/python" -c "
from tensortrade_platform.training.experiment_store import ExperimentStore
from tensortrade_platform.training.hyperparameter_store import HyperparameterStore
from tensortrade_platform.training.dataset_store import DatasetStore
store = ExperimentStore()
hp = HyperparameterStore()
ds = DatasetStore()
print(f'  Database:   $db_path')
print(f'  Datasets:   {len(ds.list_datasets())}')
print(f'  HP packs:   {len(hp.list_packs())}')
print(f'  Experiments: {len(store.list_experiments())}')
store.close(); hp.close(); ds.close()
"
    echo -e "${GREEN}Database ready.${NC}"
}

case "${1:-start}" in
    start)    cmd_start ;;
    stop)     cmd_stop ;;
    status)   cmd_status ;;
    reset-db) cmd_reset_db ;;
    init-db)  cmd_init_db ;;
    *)
        echo "Usage: ./dev.sh [start|stop|status|init-db|reset-db]"
        exit 1
        ;;
esac
