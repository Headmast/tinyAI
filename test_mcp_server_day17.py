"""Quick test: verify MCP server tools work end-to-end."""
import subprocess, sys, json, re, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

proc = subprocess.Popen(
    [sys.executable, "mcp_server.py"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, bufsize=1
)

msg_counter = 0

def send_recv(method, params=None):
    global msg_counter
    msg_counter += 1
    msg = {"jsonrpc": "2.0", "id": msg_counter, "method": method}
    if params:
        msg["params"] = params
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()
    raw = proc.stdout.readline()
    return json.loads(raw)

# 1. Initialize
r = send_recv("initialize", {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {"name": "test", "version": "1.0"}
})
info = r.get("result", {}).get("serverInfo", {})
print(f"1. initialize: {info}")
assert info.get("name") == "tinyai-logs-server"

# 2. tools/list
r = send_recv("tools/list")
tools = r.get("result", {}).get("tools", [])
names = [t["name"] for t in tools]
print(f"2. tools/list: {len(tools)} tools: {names}")
assert len(tools) == 9
assert "save_memory" in names
assert "delete_memory_key" in names
assert "get_conversation_summary" in names

# 3. save_memory
r = send_recv("tools/call", {"name": "save_memory", "arguments": {
    "filename": "test_day17.json",
    "key": "demo_key",
    "value": "hello from day 17"
}})
text = r["result"]["content"][0]["text"]
print(f"3. save_memory: {text}")
assert "Сохранено" in text

# 4. read_memory
r = send_recv("tools/call", {"name": "read_memory", "arguments": {
    "filename": "test_day17.json"
}})
text = r["result"]["content"][0]["text"]
data = json.loads(text)
print(f"4. read_memory: demo_key = {data.get('demo_key')}")
assert data["demo_key"] == "hello from day 17"

# 5. delete_memory_key
r = send_recv("tools/call", {"name": "delete_memory_key", "arguments": {
    "filename": "test_day17.json",
    "key": "demo_key"
}})
text = r["result"]["content"][0]["text"]
print(f"5. delete_memory_key: {text}")
assert "Удалено" in text

# 6. get_conversation_summary
r = send_recv("tools/call", {"name": "list_logs", "arguments": {}})
logs_text = r["result"]["content"][0]["text"]
m = re.search(r"(conversation_\S+\.json)", logs_text)
if m:
    log_name = m.group(1)
    r = send_recv("tools/call", {"name": "get_conversation_summary", "arguments": {"filename": log_name}})
    text = r["result"]["content"][0]["text"]
    print(f"6. summary({log_name}): {text[:150]}")
    assert "Сводка" in text
else:
    print("6. No logs found (OK for empty workspace)")

proc.stdin.close()
proc.wait(timeout=3)

# Cleanup test file
import pathlib
test_file = pathlib.Path(__file__).parent / "memory_data" / "test_day17.json"
if test_file.exists():
    test_file.unlink()
    print("   (cleaned up test_day17.json)")

print("\n✅ All MCP server tests passed!")
