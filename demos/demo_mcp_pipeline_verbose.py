"""
Verbose демо: 4 MCP-сервера в пайплайне с полным логом всех вызовов.

Запуск:
    python demos/demo_mcp_pipeline_verbose.py "нейронной"
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SERVERS_DIR = ROOT / "pipeline" / "servers"

TOOLS_MAP = {
    "search":         SERVERS_DIR / "search_server.py",
    "summarize":      SERVERS_DIR / "summarize_server.py",
    "format_content": SERVERS_DIR / "format_server.py",
    "save_to_file":   SERVERS_DIR / "store_server.py",
    "save_to_db":     SERVERS_DIR / "store_server.py",
}

BORDER = "─" * 64


def clr(code, text):
    return f"\033[{code}m{text}\033[0m"


def dim(t):    return clr("2", t)
def cyan(t):   return clr("36", t)
def green(t):  return clr("32", t)
def yellow(t): return clr("33", t)
def bold(t):   return clr("1", t)
def red(t):    return clr("31", t)
def magenta(t):return clr("35", t)


class VerboseMCPConn:
    """MCP-соединение с полным логом каждого JSON-RPC сообщения."""

    def __init__(self, server_path: Path, pid_ref: list):
        self.server_path = server_path
        self.name = ""
        self._pid_ref = pid_ref
        self._proc = None
        self._msg_id = 0

    def start(self):
        label = self.server_path.name
        print(f"\n  {bold('▶ subprocess.Popen')}({yellow(label)})")
        self._proc = subprocess.Popen(
            [sys.executable, str(self.server_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        self._pid_ref.append(self._proc.pid)
        print(f"  {dim('PID:')} {cyan(str(self._proc.pid))}  "
              f"  {dim('транспорт:')} {cyan('stdio (JSON-RPC 2.0)')}")

    def handshake(self):
        req = {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pipeline-executor", "version": "2.0"},
            },
        }
        print(f"\n  {dim('→ SEND')} {magenta('initialize')}")
        print(f"  {dim(json.dumps(req, ensure_ascii=False))}")
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()
        raw = self._proc.stdout.readline()
        resp = json.loads(raw)
        info = resp.get("result", {}).get("serverInfo", {})
        self.name = info.get("name", "?")
        print(f"  {dim('← RECV')} {green('OK')}  "
              f"name={cyan(self.name)}  version={cyan(info.get('version','?'))}")

        note = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        self._msg_id += 1
        print(f"\n  {dim('→ SEND')} {magenta('notifications/initialized')} {dim('(no response)')}")
        self._proc.stdin.write(json.dumps(note) + "\n")
        self._proc.stdin.flush()

    def call_tool(self, name: str, arguments: dict) -> str:
        self._msg_id += 2
        req = {
            "jsonrpc": "2.0", "id": self._msg_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        args_preview = {}
        for k, v in arguments.items():
            if isinstance(v, str) and len(v) > 80:
                args_preview[k] = v[:80] + f"… ({len(v)} chars)"
            else:
                args_preview[k] = v
        print(f"\n  {dim('→ SEND')} {magenta('tools/call')}  tool={yellow(name)}")
        print(f"  {dim('  args:')} {dim(json.dumps(args_preview, ensure_ascii=False))}")
        t0 = time.time()
        self._proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        raw = self._proc.stdout.readline()
        elapsed = int((time.time() - t0) * 1000)
        resp = json.loads(raw)
        result = resp.get("result", {})
        content = result.get("content", [{}])
        text = content[0].get("text", "") if content else ""
        is_err = result.get("isError", False)
        prefix = red("← ERR") if is_err else green("← RECV")
        preview = text[:120] + ("…" if len(text) > 120 else "")
        print(f"  {prefix} {dim(f'({elapsed}мс)')}  {dim(preview)}")
        return text

    def stop(self):
        if self._proc:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
        print(f"  {dim('■ PID')} {self._proc.pid} {dim('terminated')}")


def run_verbose(query: str, platform: str = "telegram", style: str = "brief"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = query[:30].replace(" ", "_").lower()
    filename = f"{safe}_{timestamp}.md"

    print()
    print(bold("═" * 64))
    print(bold("  TinyAI — 4-серверный MCP-пайплайн (verbose)"))
    print(bold("═" * 64))
    print(f"  Запрос:    {cyan(query)}")
    print(f"  Платформа: {cyan(platform)}")
    print(f"  Стиль:     {cyan(style)}")
    print(f"  Время:     {dim(timestamp)}")

    pids = []
    conns = {}         # path_str → VerboseMCPConn
    step_results = []  # (tool_name, raw_text)
    pipeline_start = time.time()

    steps = [
        ("search",         {"query": query, "limit": 5},              "Search   [search_server]"),
        ("summarize",      {"text": "__PREV__", "style": style},       "Summarize [summarize_server]"),
        ("format_content", {"content": "__PREV__", "platform": platform, "title": query},
                                                                        "Format   [format_server]"),
        ("save_to_file",   {"content": "__PREV__", "filename": filename}, "SaveFile [store_server]"),
        ("save_to_db",     {"title": query, "content": "__STEP2__",
                            "platform": platform, "tags": [style, platform]},
                                                                        "SaveDB   [store_server]"),
    ]

    total = len(steps)

    for i, (tool, args, label) in enumerate(steps):
        print()
        print(f"{bold(BORDER)}")
        print(f"{bold(f'  Шаг {i+1}/{total}:')} {yellow(label)}")
        print(BORDER)

        # подстановка
        if "__PREV__" in str(args):
            prev_text = step_results[-1][1] if step_results else ""
            args = {k: (prev_text if v == "__PREV__" else v) for k, v in args.items()}
        if "__STEP2__" in str(args):
            step2_text = step_results[2][1] if len(step_results) > 2 else ""
            args = {k: (step2_text if v == "__STEP2__" else v) for k, v in args.items()}

        # Распарсить $prev.result из summarize — нужен текст снипетов
        if tool == "summarize" and step_results:
            raw_search = step_results[0][1]
            try:
                data = json.loads(raw_search)
                snippets = "\n\n".join(
                    r["snippet"] for r in data.get("results", [])
                )
                args["text"] = snippets or raw_search
            except Exception:
                args["text"] = raw_search

        server_path = TOOLS_MAP[tool]
        key = str(server_path)
        if key not in conns:
            print(f"\n{dim('  [новый MCP-сервер]')}")
            conn = VerboseMCPConn(server_path, pids)
            conn.start()
            conn.handshake()
            conns[key] = conn
        else:
            print(f"\n  {dim('[переиспользуем')} {cyan(conns[key].name)}{dim(']')}")

        raw = conns[key].call_tool(tool, args)
        step_results.append((tool, raw))

        # Краткая сводка шага
        try:
            d = json.loads(raw)
            if "found" in d:
                print(f"\n  {green('✓')} Найдено: {bold(str(d['found']))} фрагментов")
            elif "summary_length" in d:
                print(f"\n  {green('✓')} Сжато: {d['original_length']} → {bold(str(d['summary_length']))} символов")
            elif "char_count" in d:
                print(f"\n  {green('✓')} Формат {bold(d['platform'])}: {bold(str(d['char_count']))} символов")
            elif "path" in d:
                print(f"\n  {green('✓')} Файл: {cyan(d['path'])}")
            elif "content_id" in d:
                cid = d['content_id']
                print(f"\n  {green('✓')} БД: {bold(f'ID={cid}')}  status={d.get('status','?')}")
        except Exception:
            pass

    total_ms = int((time.time() - pipeline_start) * 1000)

    # Завершение серверов
    print()
    print(f"{bold(BORDER)}")
    print(bold("  Завершение серверов"))
    print(BORDER)
    for conn in conns.values():
        conn.stop()

    # Итог
    print()
    print(f"{bold('═' * 64)}")
    print(f"{bold(green('  ✅ Пайплайн завершён'))}  {dim(f'{total_ms / 1000:.1f}с')}")
    print(f"  Запущено процессов: {bold(str(len(set(pids))))}")
    print(f"  Серверов:           {bold(str(len(conns)))}")

    # финальный контент
    if len(step_results) >= 3:
        try:
            formatted = json.loads(step_results[2][1]).get("formatted", "")
            print()
            print(f"  {bold('─── Готовый пост ─────────────────────────')}")
            for line in formatted.splitlines():
                print(f"  {line}")
        except Exception:
            pass
    print(f"{bold('═' * 64)}")


if __name__ == "__main__":
    import argparse, os
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Поисковый запрос / тема")
    parser.add_argument("--platform", default="telegram", choices=["telegram", "website", "rss", "plain"])
    parser.add_argument("--style", default="brief", choices=["news", "digest", "brief"])
    args = parser.parse_args()

    run_verbose(args.query, args.platform, args.style)
