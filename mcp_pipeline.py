"""
Pipeline Executor — 4-серверный MCP-пайплайн (День 19).

Архитектура: 4 отдельных MCP-сервера, каждый — свой subprocess:
  Server 1  search_server    → search
  Server 2  summarize_server → summarize
  Server 3  format_server    → format_content
  Server 4  store_server     → save_to_file, save_to_db

Исполнитель запускает все 4 сервера, выполняет MCP handshake с каждым,
затем маршрутизирует вызовы инструментов к нужному серверу.

Подстановки:
  $prev.result        — результат предыдущего шага
  $steps.N.result     — результат шага N (0-based)
  $steps.N.json.field — JSON-поле из результата шага N

Пример:
    executor = PipelineExecutor()
    result = executor.run([
        Step(tool="search", args={"query": "AI"}),
        Step(tool="summarize", args={"text": "$prev.result"}),
        Step(tool="format_content", args={"content": "$prev.result", "platform": "telegram"}),
        Step(tool="save_to_file", args={"content": "$prev.result", "filename": "out.md"}),
        Step(tool="save_to_db", args={"title": "AI", "content": "$steps.2.result", "platform": "telegram"}),
    ])
"""

import json
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

SERVERS_DIR = Path(__file__).parent / "pipeline" / "servers"

# Маппинг: инструмент → файл сервера
TOOL_TO_SERVER = {
    "search": SERVERS_DIR / "search_server.py",
    "summarize": SERVERS_DIR / "summarize_server.py",
    "format_content": SERVERS_DIR / "format_server.py",
    "save_to_file": SERVERS_DIR / "store_server.py",
    "save_to_db": SERVERS_DIR / "store_server.py",
}


@dataclass
class Step:
    """Один шаг пайплайна."""
    tool: str
    args: Dict[str, Any]
    name: Optional[str] = None


@dataclass
class StepResult:
    """Результат выполнения одного шага."""
    step_index: int
    tool: str
    server: str          # имя MCP-сервера, который обработал вызов
    raw: str
    parsed: Any = None
    error: Optional[str] = None
    elapsed_ms: int = 0


@dataclass
class PipelineResult:
    """Итог выполнения всего пайплайна."""
    run_id: str
    steps: List[StepResult] = field(default_factory=list)
    status: str = "completed"
    error: Optional[str] = None
    total_ms: int = 0
    servers_used: List[str] = field(default_factory=list)


class _MCPConnection:
    """Одно stdio-соединение с MCP-сервером."""

    def __init__(self, server_path: Path, verbose: bool = False):
        self.server_path = server_path
        self.name = ""
        self.version = ""
        self._proc: Optional[subprocess.Popen] = None
        self._msg_id = 0
        self._verbose = verbose

    def start(self):
        self._proc = subprocess.Popen(
            [sys.executable, str(self.server_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def handshake(self):
        resp = self._send_recv("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "pipeline-executor", "version": "2.0"},
        })
        info = resp.get("result", {}).get("serverInfo", {})
        self.name = info.get("name", "?")
        self.version = info.get("version", "?")
        # notification
        self._msg_id += 1
        note = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        self._proc.stdin.write(json.dumps(note) + "\n")
        self._proc.stdin.flush()

    def call_tool(self, name: str, arguments: dict) -> str:
        resp = self._send_recv("tools/call", {"name": name, "arguments": arguments})
        result = resp.get("result", {})
        if result.get("isError"):
            content = result.get("content", [{}])
            return content[0].get("text", "unknown error") if content else "unknown error"
        content = result.get("content", [{}])
        return content[0].get("text", "") if content else ""

    def stop(self):
        if self._proc:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None

    def _send_recv(self, method: str, params: Optional[dict] = None) -> dict:
        self._msg_id += 1
        msg = {"jsonrpc": "2.0", "id": self._msg_id, "method": method}
        if params:
            msg["params"] = params
        self._proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        raw = self._proc.stdout.readline()
        if not raw:
            raise RuntimeError(f"MCP server {self.server_path.name} closed unexpectedly")
        return json.loads(raw)


class PipelineExecutor:
    """
    Исполнитель 4-серверного MCP-пайплайна.

    Каждый инструмент маршрутизируется к своему MCP-серверу.
    Серверы запускаются лениво (при первом обращении) и переиспользуются.
    """

    def __init__(self, verbose: bool = True, save_to_db: bool = True):
        self._verbose = verbose
        self._save_to_db = save_to_db
        self._connections: Dict[str, _MCPConnection] = {}

    def run(self, steps: List[Step]) -> PipelineResult:
        run_id = str(uuid.uuid4())[:8]
        result = PipelineResult(run_id=run_id)
        start = time.time()

        try:
            if self._save_to_db:
                self._record_run_start(run_id, steps)

            for i, step in enumerate(steps):
                resolved_args = self._resolve_args(step.args, result.steps)
                step_label = step.name or step.tool
                self._log(f"[{i + 1}/{len(steps)}] {step_label}...")

                conn = self._get_connection(step.tool)
                server_name = conn.name

                t0 = time.time()
                raw = conn.call_tool(step.tool, resolved_args)
                elapsed = int((time.time() - t0) * 1000)

                parsed = None
                error = None
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict) and parsed.get("error"):
                        error = parsed["error"]
                except (json.JSONDecodeError, TypeError):
                    pass

                sr = StepResult(
                    step_index=i,
                    tool=step.tool,
                    server=server_name,
                    raw=raw,
                    parsed=parsed,
                    error=error,
                    elapsed_ms=elapsed,
                )
                result.steps.append(sr)

                if server_name not in result.servers_used:
                    result.servers_used.append(server_name)

                if error:
                    result.status = "failed"
                    result.error = f"Step {i} ({step.tool}@{server_name}): {error}"
                    self._log(f"  ✗ ошибка: {error}")
                    break

                self._log_step_result(sr)

        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            self._log(f"✗ Ошибка пайплайна: {e}")
        finally:
            result.total_ms = int((time.time() - start) * 1000)
            self._stop_all()

            if self._save_to_db:
                self._record_run_finish(run_id, result)

        if result.status == "completed":
            self._log(f"✅ Пайплайн завершён за {result.total_ms / 1000:.1f}с")
            self._log(f"   Серверы: {', '.join(result.servers_used)}")
        return result

    # ── Server management ─────────────────────────────────────────────────────

    def _get_connection(self, tool_name: str) -> _MCPConnection:
        server_path = TOOL_TO_SERVER.get(tool_name)
        if not server_path:
            raise ValueError(f"Unknown tool: {tool_name}")

        key = str(server_path)
        if key not in self._connections:
            conn = _MCPConnection(server_path, verbose=self._verbose)
            conn.start()
            conn.handshake()
            self._log(f"  🔗 {conn.name} v{conn.version}")
            self._connections[key] = conn
        return self._connections[key]

    def _stop_all(self):
        for conn in self._connections.values():
            conn.stop()
        self._connections.clear()

    # ── Подстановки ───────────────────────────────────────────────────────────

    def _resolve_args(
        self, args: Dict[str, Any], completed: List[StepResult]
    ) -> Dict[str, Any]:
        resolved = {}
        for key, value in args.items():
            if isinstance(value, str):
                resolved[key] = self._resolve_value(value, completed)
            elif isinstance(value, list):
                resolved[key] = [
                    self._resolve_value(v, completed) if isinstance(v, str) else v
                    for v in value
                ]
            else:
                resolved[key] = value
        return resolved

    def _resolve_value(self, value: str, completed: List[StepResult]) -> str:
        if "$prev.result" in value and completed:
            prev = completed[-1]
            replacement = self._extract_main_text(prev)
            value = value.replace("$prev.result", replacement)

        pattern = re.compile(r"\$steps\.(\d+)\.result")
        for m in pattern.finditer(value):
            idx = int(m.group(1))
            if 0 <= idx < len(completed):
                replacement = self._extract_main_text(completed[idx])
                value = value.replace(m.group(0), replacement)

        pattern_json = re.compile(r"\$steps\.(\d+)\.json\.(\w+)")
        for m in pattern_json.finditer(value):
            idx = int(m.group(1))
            field_name = m.group(2)
            if 0 <= idx < len(completed) and completed[idx].parsed:
                field_val = completed[idx].parsed.get(field_name, "")
                replacement = str(field_val) if not isinstance(field_val, str) else field_val
                value = value.replace(m.group(0), replacement)

        return value

    def _extract_main_text(self, step_result: StepResult) -> str:
        if step_result.parsed and isinstance(step_result.parsed, dict):
            for key in ("summary", "formatted", "content"):
                if key in step_result.parsed:
                    val = step_result.parsed[key]
                    if isinstance(val, str):
                        return val
            if "results" in step_result.parsed:
                val = step_result.parsed["results"]
                if isinstance(val, list):
                    snippets = []
                    for item in val:
                        if isinstance(item, dict) and "snippet" in item:
                            snippets.append(item["snippet"])
                        elif isinstance(item, str):
                            snippets.append(item)
                    if snippets:
                        return "\n\n".join(snippets)
                    return json.dumps(val, ensure_ascii=False)
                if isinstance(val, str):
                    return val
            for key in ("saved", "path"):
                if key in step_result.parsed:
                    val = step_result.parsed[key]
                    if isinstance(val, str):
                        return val
                    return json.dumps(val, ensure_ascii=False)
        return step_result.raw

    # ── DB tracking ───────────────────────────────────────────────────────────

    def _record_run_start(self, run_id: str, steps: List[Step]):
        try:
            from pipeline.db import init_db, create_pipeline_run
            init_db()
            steps_data = [{"tool": s.tool, "args": s.args, "name": s.name} for s in steps]
            create_pipeline_run(run_id, steps_data)
        except Exception:
            pass

    def _record_run_finish(self, run_id: str, result: PipelineResult):
        try:
            from pipeline.db import finish_pipeline_run
            results_data = [
                {
                    "tool": sr.tool,
                    "server": sr.server,
                    "elapsed_ms": sr.elapsed_ms,
                    "error": sr.error,
                    "raw_length": len(sr.raw),
                }
                for sr in result.steps
            ]
            finish_pipeline_run(
                run_id, results=results_data,
                status=result.status, error=result.error,
                total_ms=result.total_ms,
            )
        except Exception:
            pass

    # ── Логирование ───────────────────────────────────────────────────────────

    def _log(self, msg: str):
        if self._verbose:
            print(msg)

    def _log_step_result(self, sr: StepResult):
        if not self._verbose:
            return
        if sr.parsed and isinstance(sr.parsed, dict):
            if "found" in sr.parsed:
                self._log(f"  ✓ найдено: {sr.parsed['found']} фрагментов")
            elif "summary_length" in sr.parsed:
                orig = sr.parsed.get("original_length", "?")
                summ = sr.parsed.get("summary_length", "?")
                self._log(f"  ✓ сжато: {orig} → {summ} символов")
            elif "char_count" in sr.parsed and "platform" in sr.parsed:
                self._log(f"  ✓ формат: {sr.parsed['platform']}, {sr.parsed['char_count']} сим.")
            elif "path" in sr.parsed:
                self._log(f"  ✓ файл: {sr.parsed['path']}")
            elif "content_id" in sr.parsed:
                self._log(f"  ✓ БД: ID={sr.parsed['content_id']}, status={sr.parsed.get('status', '?')}")
            else:
                self._log(f"  ✓ ок ({sr.elapsed_ms}мс)")
        else:
            self._log(f"  ✓ ок ({sr.elapsed_ms}мс)")
