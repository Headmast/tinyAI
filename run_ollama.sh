#!/usr/bin/env bash
# run_ollama.sh — Запуск локального Ollama с моделью qwen3:8b
#
# Использование:
#   ./run_ollama.sh          — запуск (скачает модель при первом запуске)
#   ./run_ollama.sh stop     — остановить Ollama
#   ./run_ollama.sh status   — проверить статус
#   ./run_ollama.sh models   — список загруженных моделей

set -euo pipefail

MODEL="${OLLAMA_MODEL:-qwen3:8b}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"

export OLLAMA_HOST

_info()  { printf "\033[36m→ %s\033[0m\n" "$*"; }
_ok()    { printf "\033[32m✓ %s\033[0m\n" "$*"; }
_err()   { printf "\033[31m✗ %s\033[0m\n" "$*" >&2; }

check_ollama_installed() {
    if ! command -v ollama &>/dev/null; then
        _err "Ollama не найден. Установите: https://ollama.com/download"
        exit 1
    fi
}

is_running() {
    curl -sf "http://${OLLAMA_HOST}/api/tags" &>/dev/null
}

start_server() {
    if is_running; then
        _ok "Ollama уже запущен на ${OLLAMA_HOST}"
        return 0
    fi

    _info "Запускаем Ollama сервер на ${OLLAMA_HOST}..."
    ollama serve &>/dev/null &
    OLLAMA_PID=$!

    # Ждём готовности (до 15 секунд)
    for i in $(seq 1 30); do
        if is_running; then
            _ok "Ollama запущен (PID: ${OLLAMA_PID})"
            return 0
        fi
        sleep 0.5
    done

    _err "Не удалось запустить Ollama за 15 секунд"
    exit 1
}

pull_model() {
    _info "Проверяем модель ${MODEL}..."
    if ollama list 2>/dev/null | grep -q "${MODEL}"; then
        _ok "Модель ${MODEL} уже загружена"
    else
        _info "Скачиваем модель ${MODEL} (может занять время)..."
        ollama pull "${MODEL}"
        _ok "Модель ${MODEL} загружена"
    fi
}

stop_server() {
    _info "Останавливаем Ollama..."
    pkill -f "ollama serve" 2>/dev/null && _ok "Ollama остановлен" || _err "Ollama не запущен"
}

show_status() {
    if is_running; then
        _ok "Ollama работает на ${OLLAMA_HOST}"
        echo ""
        _info "Загруженные модели:"
        ollama list 2>/dev/null || echo "  (нет моделей)"
    else
        _err "Ollama не запущен"
    fi
}

case "${1:-start}" in
    start)
        check_ollama_installed
        start_server
        pull_model
        echo ""
        _ok "Готово! Ollama + ${MODEL} доступны"
        _info "API endpoint: http://${OLLAMA_HOST}/v1"
        _info "Используйте в TinyAI: model qwen3:8b"
        echo ""
        _info "Для остановки: ./run_ollama.sh stop"
        # Держим процесс живым (для удобства)
        if [[ -n "${OLLAMA_PID:-}" ]]; then
            _info "Нажмите Ctrl+C для остановки..."
            wait "${OLLAMA_PID}" 2>/dev/null || true
        fi
        ;;
    stop)
        stop_server
        ;;
    status)
        show_status
        ;;
    models)
        ollama list 2>/dev/null || _err "Ollama не запущен"
        ;;
    *)
        echo "Использование: $0 {start|stop|status|models}"
        exit 1
        ;;
esac
