# ChannelStat: как запустить скрипт ChannelMessages

Полный набор команд для запуска скрипта с профилями ChannelStat.

## Аутентификация: вход по Telegram ссылке

При первом запуске скрипт выведет специальную Telegram-ссылку для входа:

```
tg://login?token=AQJOevhpTpU1Snv83m0p9s3isH9BQhrN-yAFivRdad
```

**Как войти:**
1. **На мобильном**: кликните на ссылку → Telegram откроется → нажмите "Confirm" → готово!
2. **На компьютере**: скопируйте ссылку, откройте Telegram, вставьте в address bar → подтвердите вход
3. Вернитесь в консоль и нажмите **Enter**

> 💡 Это самый надежный способ — не требует кодов, QR-сканирования или ожидания уведомлений!

---

## 1) Перейти в нужную папку

```bash
cd /Users/kirillklebanov/Documents/Code/AIcourse/TinyAI/tinyAI/ChannelMessages
```

## 2) Активировать Python-окружение проекта

```bash
source ../.venv/bin/activate
```

## 3) Установить зависимости (один раз)

```bash
pip install -r requirements.txt
```

## 4) Запуск скрипта

```bash
cd /Users/kirillklebanov/Documents/Code/AIcourse/TinyAI/tinyAI/ChannelMessages
/Users/kirillklebanov/Documents/Code/AIcourse/TinyAI/tinyAI/.venv/bin/python main.py
```

## 5) Если нужно пересоздать сессию

```bash
cd /Users/kirillklebanov/Documents/Code/AIcourse/TinyAI/tinyAI/ChannelMessages
rm -f channel_stats.session channel_stats.session-journal
/Users/kirillklebanov/Documents/Code/AIcourse/TinyAI/tinyAI/.venv/bin/python main.py
```

> При следующем запуске скрипт попросит новый код из Telegram

## 6) Проверка таблиц в SQLite

```bash
cd /Users/kirillklebanov/Documents/Code/AIcourse/TinyAI/tinyAI/ChannelMessages
sqlite3 channel_stats.db ".tables"
```

## 7) Проверка, что session авторизована

```bash
cd /Users/kirillklebanov/Documents/Code/AIcourse/TinyAI/tinyAI/ChannelMessages
/Users/kirillklebanov/Documents/Code/AIcourse/TinyAI/tinyAI/.venv/bin/python - <<'PY'
import asyncio
from telethon import TelegramClient
from dotenv import dotenv_values

env = dotenv_values('.env')
api_id = int(env.get('API_ID') or 0)
api_hash = env.get('API_HASH') or ''
session = env.get('SESSION_NAME', 'channel_stats')

async def main():
    c = TelegramClient(session, api_id, api_hash)
    await c.connect()
    ok = await c.is_user_authorized()
    await c.disconnect()
    print(f'Session {session}: authorized={ok}')

asyncio.run(main())
PY
```


