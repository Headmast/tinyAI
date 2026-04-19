"""
Минимальный тест: один API вызов GLM-4.7, полный вывод промпта и reasoning.
Запуск: python3 -u test_api_log.py
"""
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

W = 70

def p(msg="", flush=True):
    print(msg, flush=flush)

def hr(c="─"):
    p(c * W)

# ── 1. Конфигурация ──────────────────────────────────────────────────
p()
hr("═")
p("  ТЕСТ ЛОГИРОВАНИЯ API → GLM-4.7")
hr("═")

API_KEY  = os.getenv("CLOUD_API_KEY", "")
BASE_URL = "https://foundation-models.api.cloud.ru/v1"
MODEL    = "zai-org/GLM-4.7"

p(f"  API_KEY : {'SET (' + API_KEY[:8] + '...)' if API_KEY else 'НЕ ЗАДАН!'}")
p(f"  BASE_URL: {BASE_URL}")
p(f"  MODEL   : {MODEL}")
hr("═")

if not API_KEY:
    p("❌ CLOUD_API_KEY не задан в .env — выходим")
    sys.exit(1)

# ── 2. Клиент ────────────────────────────────────────────────────────
client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120)

# ── 3. Промпт ────────────────────────────────────────────────────────
SYSTEM = "Ты — журналист. Отвечай кратко и по делу."
USER   = "Напиши заметку в 3 предложения: почему Python популярен в 2025."

p()
hr("▼")
p("  📤 ЗАПРОС К API")
hr("▼")
p(f"  system : {SYSTEM}")
p(f"  user   : {USER}")
p(f"  model  : {MODEL}")
p(f"  max_completion_tokens: 2000")
hr("▼")
p("  ⏳ Ожидаем ответ...", flush=True)

# ── 4. Вызов ─────────────────────────────────────────────────────────
t0 = time.time()
try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": USER},
        ],
        max_completion_tokens=2000,
        temperature=0.5,
    )
    elapsed = time.time() - t0
except Exception as exc:
    p(f"\n  ❌ API ОШИБКА: {exc}")
    sys.exit(1)

# ── 5. Разбор ответа ─────────────────────────────────────────────────
msg       = response.choices[0].message
content   = msg.content or ""
reasoning = getattr(msg, "reasoning", None) or ""
usage     = response.usage
finish    = response.choices[0].finish_reason

p()
hr("▲")
p(f"  📥 ОТВЕТ API  ({elapsed:.1f}с)")
hr("▲")
p(f"  finish_reason : {finish}")
p(f"  токены        : prompt={usage.prompt_tokens}  completion={usage.completion_tokens}  total={usage.total_tokens}")
hr()

p(f"  💭 REASONING ({len(reasoning)} симв.):")
hr()
if reasoning:
    for line in reasoning.split("\n"):
        p(f"  {line}")
else:
    p("  (пусто — возможно не хватило токенов)")
hr()

p(f"  📝 CONTENT ({len(content)} симв.):")
hr()
if content:
    for line in content.split("\n"):
        p(f"  {line}")
else:
    p("  (пусто — возможно не хватило токенов)")
hr("▲")

p()
if content:
    p("  ✅ Логирование работает корректно")
else:
    p("  ⚠️  content пуст, reasoning есть — нужно больше токенов")
p()
