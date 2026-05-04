# Telegram Channel Stats Collector

Скрипт под обычными кредами **пользователя** (не бота): проходит по всем broadcast-каналам, в которых вы состоите, собирает последние 10 сообщений с каждого, считает статистику и сохраняет в SQLite (или PostgreSQL).

---

## Что собирается

| Что | Где хранится |
|-----|-------------|
| Список каналов | таблица `channels` |
| Последние 10 сообщений каждого канала | таблица `messages` |
| Статистика на момент запуска | таблица `channel_stats` |

**Статистика по каналу** (на последние 10 сообщений):
- количество сообщений
- средняя длина текста (в символах)
- количество сообщений с медиа
- самый активный отправитель

---

## Установка

В текущем репозитории Python-окружение уже находится уровнем выше: `tinyAI/.venv`.

```bash
cd tinyAI/ChannelMessages
source ../.venv/bin/activate    # Windows: ..\.venv\Scripts\activate
pip install -r requirements.txt
```

Если хотите отдельное окружение именно для этого подпроекта, создайте его вручную:

```bash
cd tinyAI/ChannelMessages
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Настройка credentials

### 1. Получите API_ID и API_HASH

1. Откройте [https://my.telegram.org](https://my.telegram.org)
2. Войдите своим номером телефона
3. Выберите **API development tools**
4. Создайте приложение (название и платформа — любые)
5. Скопируйте **App api_id** и **App api_hash**

### 2. Создайте `.env` файл

```bash
cp .env.example .env
```

Откройте `.env` и заполните реальными значениями:

```env
API_ID=12345678
API_HASH=0123456789abcdef0123456789abcdef
PHONE_NUMBER=+79001234567
SESSION_NAME=channel_stats
DB_URL=sqlite+aiosqlite:///channel_stats.db
```

> **Важно**: `.env` уже добавлен в `.gitignore`. Никогда не коммитьте этот файл в git.

### Переменные окружения

| Переменная | Описание | Обязательна |
|-----------|----------|-------------|
| `API_ID` | ID приложения с my.telegram.org | ✅ |
| `API_HASH` | Hash приложения с my.telegram.org | ✅ |
| `PHONE_NUMBER` | Ваш номер телефона (с кодом страны) | ✅ (только при первом запуске) |
| `SESSION_NAME` | Имя файла сессии (без расширения) | нет, по умолчанию `channel_stats` |
| `DB_URL` | URL базы данных | нет, по умолчанию SQLite |
| `MESSAGES_LIMIT` | Количество сообщений с канала | нет, по умолчанию `10` |

---

## Запуск

```bash
python main.py
```

### Первый запуск — Вход по Telegram ссылке

При первом запуске скрипт покажет специальную ссылку для входа:

```
======================================================================
TELEGRAM LOGIN
======================================================================

🔗 Click the link below (or copy and paste in Telegram):

   tg://login?token=AQJOevhpTpU1Snv83m0p9s3isH9BQhrN-yAFivRdad

Alternative: copy the link above, paste in Telegram address bar
======================================================================

Press Enter after you've authenticated in Telegram: 
```

### Как войти

**Вариант 1: Кликнуть по ссылке (на мобильном)**
1. Если вы на мобильном устройстве — просто **кликните на ссылку**
2. Telegram откроется и попросит подтвердить вход
3. Нажмите "**Log in**" или "**Confirm**"
4. Готово! Вернитесь в консоль и нажмите Enter

**Вариант 2: Скопировать ссылку (на компьютере)**
1. **Скопируйте полную ссылку** из консоли
2. Откройте **Telegram** на телефоне
3. Перейдите в **Settings → Devices → Scan QR code** (или подобное)
4. **Вставьте ссылку** в адресную строку Telegram (как при открытии ссылки)
5. Подтвердите вход — готово!

> 💡 Это самый надежный способ входа — не требует ввода кодов, QR-сканирования или ожидания уведомлений!

### Последующие запуски

После первого входа скрипт сохраняет сессию в файл (например `channel_stats.session`). При следующих запусках никакого входа не требуется — сессия восстанавливается автоматически.

После успешного входа создаётся файл `channel_stats.session` — это сохранённая сессия.

### Повторные запуски

Файл `.session` уже существует — авторизация не требуется, скрипт запускается сразу.

---

## Полный сценарий использования

1. Установите зависимости и заполните `.env`.
2. Запустите `python main.py`.
3. На первом запуске подтвердите вход кодом из Telegram.
4. Дождитесь окончания обхода каналов и сохранения в БД.
5. Проверьте результаты через `sqlite3` (или любой SQL-клиент).

Минимальный ежедневный цикл:

```bash
cd tinyAI/ChannelMessages
source ../.venv/bin/activate
python main.py
```

Рекомендация: запускать по cron/launchd 1-2 раза в день, если нужна история динамики в `channel_stats`.

### Пример вывода

```
Initialising database …
Connecting to Telegram …

================================================================
  Channel Stats — 12 channels collected
================================================================
Channel                             Msgs Avg chars Media  Top sender
----------------------------------------------------------------
Dev News                              10      45.2     3  Channel Admin
Python Weekly                         10     312.7     8  Channel Admin
TechMemes                              10      12.1     9  Anonymous
================================================================
Results saved to the database.
```

---

## Просмотр результатов

### SQLite (командная строка)

```bash
sqlite3 channel_stats.db

-- Последняя статистика по каналам
SELECT c.title, s.collected_at, s.msg_count, s.avg_char_length, s.media_count, s.top_sender_name
FROM channel_stats s
JOIN channels c ON c.tg_id = s.channel_tg_id
ORDER BY s.collected_at DESC;

-- Сообщения конкретного канала
SELECT m.date, m.sender_name, m.text, m.has_media
FROM messages m
JOIN channels c ON c.tg_id = m.channel_tg_id
WHERE c.title = 'Dev News'
ORDER BY m.date DESC;
```

### PostgreSQL (опционально)

Установите `psycopg` и смените DB_URL в `.env`:

```bash
pip install "psycopg[binary]"
```

```env
DB_URL=postgresql+psycopg://user:password@localhost:5432/tg_stats
```

---

## Проверки и тесты

Ниже практичный checklist, чтобы убедиться, что всё работает корректно.

### Smoke test (обязательный)

```bash
cd tinyAI/ChannelMessages
source ../.venv/bin/activate
python main.py
```

Ожидаемый результат:
- скрипт отрабатывает без traceback
- появляется/обновляется файл `channel_stats.db`
- в консоли есть строка `Channel Stats — N channels collected`

### Проверка структуры БД

```bash
sqlite3 channel_stats.db ".tables"
```

Должны быть таблицы:
- `channels`
- `messages`
- `channel_stats`

### Проверка, что данные реально пишутся

```bash
sqlite3 channel_stats.db "SELECT COUNT(*) FROM channels;"
sqlite3 channel_stats.db "SELECT COUNT(*) FROM messages;"
sqlite3 channel_stats.db "SELECT COUNT(*) FROM channel_stats;"
```

После первого успешного запуска все три значения должны быть больше нуля (если у пользователя есть broadcast-каналы).

### Проверка повторного запуска (регресс)

Повторно выполните:

```bash
python main.py
```

Ожидаемый результат:
- повторного запроса кода подтверждения нет
- не возникает ошибок по уникальности сообщений
- в `channel_stats` добавляется новый snapshot

---

## Быстрая диагностика проблем

### Ошибка `API_ID/API_HASH invalid`

Проверьте значения в `.env`. Они должны быть именно с [my.telegram.org](https://my.telegram.org), а не bot token.

### Скрипт просит код на каждом запуске

Проверьте, что файл `*.session` создаётся в папке проекта и не удаляется между запусками.

### Код не приходит по SMS

Для этого сценария Telegram может возвращать тип доставки `SentCodeTypeApp`. Это означает, что код приходит в уже залогиненное приложение Telegram, а не в SMS. Проверьте Telegram на телефоне, Telegram Desktop и сервисные сообщения от Telegram.

Если с новым `API_ID/API_HASH` код приходит, а со старым нет, это обычно означает, что Telegram по-разному применяет политику доставки для разных client apps. В таком случае проблема не в логике `main.py`, а в конкретной паре credentials приложения. Практическое решение: создать новый client на `my.telegram.org`, обновить `.env` и заново создать `*.session`.

### В базе 0 каналов

Скрипт обрабатывает только broadcast-каналы. Если в аккаунте только группы/чаты, результат может быть пустым.

### Ошибка по SQLite-файлу

Убедитесь, что процесс имеет права записи в папку `tinyAI/ChannelMessages`.

---

## Безопасность

| Файл | Что содержит | Коммитить? |
|------|-------------|-----------|
| `.env` | API_ID, API_HASH, PHONE_NUMBER | ❌ (`*.env` в `.gitignore`) |
| `*.session` | Токен активной сессии Telegram | ❌ (`*.session` в `.gitignore`) |
| `.env.example` | Шаблон с заглушками | ✅ |
| `*.db` | База данных с собранными данными | ❌ (по усмотрению) |

> Если файл `.session` попал в git — отзовите сессию в Telegram: **Настройки → Устройства → Завершить сессию**.

---

## Структура файлов

```
ChannelMessages/
├── .env.example      # Шаблон переменных (коммитить)
├── .env              # Реальные креды (НЕ коммитить!)
├── .gitignore
├── requirements.txt
├── config.py         # Загрузка настроек из .env
├── models.py         # SQLAlchemy ORM модели
├── db.py             # Подключение к БД
├── collector.py      # Логика обхода каналов и сбора данных
└── main.py           # Точка входа
```
