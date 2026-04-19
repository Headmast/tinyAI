.PHONY: lint format test test-unit test-cov run-cli run-journalist help

help:  ## Показать справку
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

lint:  ## Проверка кода (ruff + mypy)
	python3 -m ruff check .
	python3 -m mypy news_agent/ rag/ memory_agent/ journalist_agent/ core/ --ignore-missing-imports

format:  ## Форматирование кода (ruff)
	python3 -m ruff format .
	python3 -m ruff check --fix .

test:  ## Запустить все тесты (без API)
	python3 -m pytest tests/ -v -m "not api and not integration"

test-unit:  ## Только юнит-тесты
	python3 -m pytest tests/ -v -m "unit"

test-all:  ## Все тесты включая API
	python3 -m pytest tests/ -v

test-cov:  ## Тесты с покрытием кода
	python3 -m pytest tests/ --cov=news_agent --cov=rag --cov=memory_agent \
		--cov=journalist_agent --cov=core --cov-report=html --cov-report=term-missing \
		-m "not api and not integration"
	@echo "\n→ Отчёт: open htmlcov/index.html"

run-cli:  ## Запуск главного CLI
	python3 llm_cli.py

run-journalist:  ## Запуск FSM-агента журналиста (3 темы)
	python3 demos/run_3topics.py --model zai-org/GLM-4.7

run-mcp:  ## Запуск MCP-агента
	python3 demos/demo_mcp_agent.py

install:  ## Установка зависимостей
	pip install -r requirements.txt

install-dev:  ## Установка dev-зависимостей
	pip install -e ".[dev]"
