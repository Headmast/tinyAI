"""
Демонстрация Дня 17: MCP-инструменты — полный цикл.

1. Регистрация инструмента + описание параметров
2. Возврат результата
3. Подключение к агенту + вызов + использование результата
"""

import subprocess
import sys
import json
import os

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)

SEP = "═" * 60
THIN = "━" * 60


# ══════════════════════════════════════════════════════════════
# ЧАСТЬ 1: Регистрация инструмента + описание параметров
# ══════════════════════════════════════════════════════════════

def demo_registration():
    print(f"\n{SEP}")
    print("  👉 1. РЕГИСТРАЦИЯ ИНСТРУМЕНТА + ОПИСАНИЕ ПАРАМЕТРОВ")
    print(f"{SEP}\n")

    proc = subprocess.Popen(
        [sys.executable, "mcp_server.py"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1
    )

    def rpc(method, params=None, mid=[0]):
        mid[0] += 1
        msg = {"jsonrpc": "2.0", "id": mid[0], "method": method}
        if params:
            msg["params"] = params
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()
        return json.loads(proc.stdout.readline())

    # Handshake
    r = rpc("initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "demo-day17", "version": "1.0"}
    })
    info = r["result"]["serverInfo"]
    print(f"  Сервер: {info['name']} v{info['version']}")
    print(f"  Протокол: MCP {r['result']['protocolVersion']}")
    print()

    # tools/list
    r = rpc("tools/list")
    tools = r["result"]["tools"]
    print(f"  Зарегистрировано инструментов: {len(tools)}\n")

    for i, t in enumerate(tools, 1):
        name = t["name"]
        desc = t["description"][:70]
        schema = t.get("inputSchema", {})
        props = schema.get("properties", {})
        required = schema.get("required", [])

        print(f"  [{i}] {name}")
        print(f"      {desc}")
        if props:
            for p, info in props.items():
                req = " *" if p in required else ""
                print(f"        • {p}{req} ({info['type']}): {info.get('description', '')[:50]}")
        else:
            print(f"        (без параметров)")
        print()

    # ══════════════════════════════════════════════════════════
    # ЧАСТЬ 2: Вызов инструмента → возврат результата
    # ══════════════════════════════════════════════════════════

    print(f"{SEP}")
    print("  👉 2. ВЫЗОВ ИНСТРУМЕНТА → ВОЗВРАТ РЕЗУЛЬТАТА")
    print(f"{SEP}\n")

    # --- save_memory ---
    print("  → tools/call: save_memory")
    print('    args: {"filename": "demo_day17.json", "key": "user_name", "value": "Кирилл"}')
    r = rpc("tools/call", {"name": "save_memory", "arguments": {
        "filename": "demo_day17.json",
        "key": "user_name",
        "value": "Кирилл"
    }})
    text = r["result"]["content"][0]["text"]
    print(f"    ✅ Результат: {text}\n")

    # --- save_memory (second key) ---
    print("  → tools/call: save_memory")
    print('    args: {"filename": "demo_day17.json", "key": "preferences", "value": "краткие ответы"}')
    r = rpc("tools/call", {"name": "save_memory", "arguments": {
        "filename": "demo_day17.json",
        "key": "preferences",
        "value": "краткие ответы с bullet points"
    }})
    text = r["result"]["content"][0]["text"]
    print(f"    ✅ Результат: {text}\n")

    # --- read_memory ---
    print("  → tools/call: read_memory")
    print('    args: {"filename": "demo_day17.json"}')
    r = rpc("tools/call", {"name": "read_memory", "arguments": {
        "filename": "demo_day17.json"
    }})
    text = r["result"]["content"][0]["text"]
    data = json.loads(text)
    print(f"    ✅ Результат:")
    print(f"       user_name: {data['user_name']}")
    print(f"       preferences: {data['preferences']}")
    print()

    # --- get_conversation_summary ---
    print("  → tools/call: get_conversation_summary")
    r_logs = rpc("tools/call", {"name": "list_logs", "arguments": {}})
    logs_text = r_logs["result"]["content"][0]["text"]
    import re
    m = re.search(r"(conversation_\S+\.json)", logs_text)
    if m:
        log_name = m.group(1)
        print(f'    args: {{"filename": "{log_name}"}}')
        r = rpc("tools/call", {"name": "get_conversation_summary", "arguments": {"filename": log_name}})
        text = r["result"]["content"][0]["text"]
        for line in text.split("\n")[:8]:
            print(f"    {line}")
    print()

    # --- delete_memory_key ---
    print("  → tools/call: delete_memory_key")
    print('    args: {"filename": "demo_day17.json", "key": "preferences"}')
    r = rpc("tools/call", {"name": "delete_memory_key", "arguments": {
        "filename": "demo_day17.json",
        "key": "preferences"
    }})
    text = r["result"]["content"][0]["text"]
    print(f"    ✅ Результат: {text}\n")

    # --- get_usage_stats ---
    print("  → tools/call: get_usage_stats")
    r = rpc("tools/call", {"name": "get_usage_stats", "arguments": {}})
    text = r["result"]["content"][0]["text"]
    for line in text.split("\n"):
        if line.strip():
            print(f"    {line}")
    print()

    proc.stdin.close()
    proc.wait(timeout=3)

    # Cleanup
    cleanup = PROJECT_ROOT / "memory_data" / "demo_day17.json"
    if cleanup.exists():
        cleanup.unlink()


# ══════════════════════════════════════════════════════════════
# ЧАСТЬ 3: Подключение к агенту — вызов через function calling
# ══════════════════════════════════════════════════════════════

def demo_agent():
    print(f"{SEP}")
    print("  👉 3. ПОДКЛЮЧЕНИЕ К АГЕНТУ + ВЫЗОВ + РЕЗУЛЬТАТ")
    print(f"{SEP}\n")

    api_key = os.getenv("CLOUD_API_KEY")
    if not api_key:
        print("  ⚠️  CLOUD_API_KEY не найден — пропуск агентной демонстрации")
        return

    sys.path.insert(0, str(PROJECT_ROOT))
    from mcp_agent import MCPAgent

    client = OpenAI(
        api_key=api_key,
        base_url="https://foundation-models.api.cloud.ru/v1",
        timeout=120.0,
    )

    tasks = [
        "Покажи список всех файлов логов разговоров.",
        "Запомни: мой любимый формат — краткий, с bullet points.",
        "Покажи статистику использования — сколько токенов потрачено?",
    ]

    with MCPAgent(client=client, verbose=True) as agent:
        for i, task in enumerate(tasks, 1):
            print(f"\n  {'─'*56}")
            print(f"  Задача {i}/{len(tasks)}")
            print(f"  {'─'*56}")
            print(f"\n  👤 User: {task}\n")

            response = agent.chat(task)

            print(f"\n  🤖 Agent: {response[:300]}")
            if len(response) > 300:
                print("  ...")
            print(f"\n  [Токены: prompt={agent.token_usage['prompt_tokens']}, "
                  f"completion={agent.token_usage['completion_tokens']}, "
                  f"total={agent.token_usage['total_tokens']}]")
            print(f"  [MCP-вызовов: {agent.tool_calls_total}]")

    # Cleanup
    cleanup = PROJECT_ROOT / "memory_data" / "agent_notes.json"
    if cleanup.exists():
        cleanup.unlink()

    print(f"\n{SEP}")
    print("  ✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print(f"{SEP}\n")


if __name__ == "__main__":
    demo_registration()
    demo_agent()
