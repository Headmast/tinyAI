"""
demo_pipeline.py — демонстрация полного цикла News Agent Pipeline
без реального API (все ответы LLM заменены реалистичными фикстурами).

Запуск:
    python3 demo_pipeline.py
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news_agent.pipeline import NewsPipeline
from news_agent.storage import PostStorage
from news_agent.formatter import OutputFormatter

TOPIC = "ЦБ России повысил ключевую ставку до 21% — максимум за 20 лет"
POST_TYPE = "breaking"

PLAN_JSON = json.dumps({
    "post_type": "breaking",
    "angle": "Историческое решение ЦБ: ставка 21% впервые с 2003 года — что это означает для россиян",
    "tone": "urgent",
    "target_audience": "Широкая аудитория, следящая за экономикой и личными финансами",
    "word_count_target": 250,
    "key_points": [
        "ЦБ поднял ставку с 19% до 21% — рекорд за 20 лет",
        "Решение принято на фоне инфляции выше 8%",
        "Вырастут ставки по ипотеке и кредитам",
        "Вкладчики получат более высокий доход"
    ],
    "structure": ["Заголовок", "Лид (кто/что/когда)", "Причины решения", "Последствия для граждан", "Прогноз"],
    "notes_for_writer": "Акцент на практических последствиях для читателя. Избегать сложной экономической терминологии."
})

RESEARCH_JSON = json.dumps({
    "background": "Банк России с 2021 года ведёт цикл повышения ставки для сдерживания инфляции. На заседании 25 октября 2024 года ставка подняла до 21% — нового исторического максимума в современной России.",
    "key_facts": [
        "Ключевая ставка повышена с 19% до 21% (рост на 2 п.п.)",
        "Последний раз ставка была на таком уровне в начале 2000-х годов",
        "Инфляция в сентябре 2024 года составила 8,6% в годовом выражении",
        "Решение принято единогласно советом директоров ЦБ",
        "Следующее заседание по ставке запланировано на декабрь 2024"
    ],
    "statistics": [
        "Инфляция: 8,6% (сентябрь 2024)",
        "Предыдущая ставка: 19%",
        "Новая ставка: 21%",
        "Ипотечные ставки ожидаются в диапазоне 23–25%"
    ],
    "key_players": ["Банк России", "Эльвира Набиуллина (глава ЦБ)", "Правительство РФ"],
    "timeline": [
        "Июль 2023 — начало цикла повышений: ставка 8,5%",
        "Февраль 2024 — ставка 16%",
        "Сентябрь 2024 — ставка 19%",
        "25 октября 2024 — ставка 21%"
    ],
    "expert_perspectives": [
        "Аналитики ожидали повышения до 20%, решение оказалось жёстче прогнозов",
        "Экономисты предупреждают о рисках замедления деловой активности"
    ],
    "controversies": [
        "Часть экономистов считает, что высокая ставка тормозит инвестиции и рост ВВП"
    ],
    "research_confidence": "high",
    "notes": "Данные актуальны на октябрь 2024 года"
})

DRAFT_TEXT = """**ЦБ поднял ставку до 21%: рекорд за 20 лет**

Банк России 25 октября повысил ключевую ставку сразу на 2 процентных пункта — с 19% до 21%. Это максимальный уровень за всю современную историю страны.

**Почему так резко?**

Главная причина — ускорение инфляции: в сентябре она составила 8,6% в годовом выражении, что вдвое превышает целевой ориентир ЦБ в 4%. Регулятор вынужден «охлаждать» экономику через удорожание кредита.

«Инфляционное давление остаётся высоким. Требуются более жёсткие денежно-кредитные условия», — пояснила глава ЦБ Эльвира Набиуллина.

**Что изменится для россиян**

Ипотека станет дороже: ставки по рыночным программам ожидаются в диапазоне 23–25% годовых. Потребительские кредиты также подорожают.

Для вкладчиков — хорошие новости: банки уже начали поднимать ставки по депозитам. В ближайшие недели можно будет открыть вклад под 22–23% годовых.

**Что дальше**

Следующее заседание Банка России пройдёт в декабре. Аналитики допускают ещё одно повышение — если инфляция не начнёт замедляться."""

EDIT_JSON = json.dumps({
    "edited_post": """**ЦБ поднял ставку до 21%: исторический рекорд**

Банк России 25 октября повысил ключевую ставку на 2 процентных пункта — с 19% до 21%. Это максимальный уровень за всю современную историю страны: в последний раз ставка была столь высокой в начале 2000-х.

**Почему именно сейчас**

Инфляция в сентябре составила 8,6% — вдвое выше целевого ориентира ЦБ. Регулятор вынужден охлаждать экономику через удорожание кредита.

«Инфляционное давление остаётся высоким. Требуются более жёсткие денежно-кредитные условия», — заявила глава ЦБ Эльвира Набиуллина.

**Что это значит для вас**

— *Ипотека и кредиты* подорожают: рыночные ставки ожидаются на уровне 23–25% годовых.
— *Вклады* станут выгоднее: банки уже повышают ставки, ожидается 22–23% годовых.
— *Бизнес*: высокая стоимость заимствований ударит по инвестиционным планам компаний.

**Что дальше**

Следующее заседание — декабрь. Аналитики не исключают ещё одного повышения, если инфляция не замедлится.""",
    "changes_made": [
        "Усилен заголовок: добавлено 'исторический рекорд'",
        "Исправлена структура: подзаголовки сделаны вопросами для вовлечённости",
        "Добавлен список последствий через тире для удобства чтения",
        "Убран повтор 'охлаждать экономику'"
    ],
    "quality_score": 9,
    "quality_breakdown": {
        "clarity": 9,
        "accuracy": 10,
        "engagement": 8,
        "structure": 9
    },
    "editorial_notes": "Хороший баланс факта и читаемости. Цитата добавляет авторитетности."
})

SEO_JSON = json.dumps({
    "headline_variants": [
        {"type": "seo", "text": "ЦБ поднял ключевую ставку до 21%: что будет с ипотекой и вкладами"},
        {"type": "social", "text": "Ставка 21%: ЦБ сделал то, чего не ожидал рынок"},
        {"type": "telegram", "text": "🔴 ЦБ: ставка 21% — исторический максимум. Ипотека дорожает, вклады растут"},
        {"type": "email", "text": "Ключевая ставка 21%: как это изменит ваши финансы"}
    ],
    "meta_description": "Банк России повысил ключевую ставку до 21% — рекорд за 20 лет. Разбираем, как это повлияет на ипотеку, кредиты и вклады.",
    "tags": ["ЦБ", "ключевая ставка", "инфляция", "ипотека", "вклады", "экономика"],
    "hashtags": ["#ЦБ", "#ключеваяставка", "#инфляция", "#ипотека", "#экономика"],
    "keywords": ["ключевая ставка 21", "ЦБ России ставка", "ипотека ставки 2024", "вклады выгодные"],
    "suggested_channels": ["Telegram", "ВКонтакте", "Яндекс.Дзен"],
    "best_posting_time": "Утро (9:00–11:00) или вечер (18:00–20:00) в будние дни"
})


def make_mock_client():
    """Создаёт mock-клиент с реалистичными заготовленными ответами."""
    client = MagicMock()
    responses = [PLAN_JSON, RESEARCH_JSON, DRAFT_TEXT, EDIT_JSON, SEO_JSON]
    call_idx = [0]

    def stream_side_effect(**kwargs):
        idx = call_idx[0]
        call_idx[0] += 1
        text = responses[idx] if idx < len(responses) else "{}"

        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = text
        chunk.choices[0].delta.reasoning_content = None
        return iter([chunk])

    client.chat.completions.create.side_effect = stream_side_effect
    return client


def main():
    print("\n" + "█" * 65)
    print("  NEWS AGENT — ДЕМОНСТРАЦИЯ ПОЛНОГО PIPELINE")
    print("  (используются реалистичные фикстуры вместо реального API)")
    print("█" * 65)
    print(f"\n  Тема: {TOPIC}")
    print(f"  Тип:  {POST_TYPE}")

    tmp_dir = tempfile.mkdtemp()
    storage = PostStorage(base_dir=tmp_dir)
    client = make_mock_client()

    pipeline = NewsPipeline(client=client, model="zai-org/GLM-4.7-Flash", verbose=True)

    result = pipeline.run(topic=TOPIC, post_type=POST_TYPE)
    result["model"] = "zai-org/GLM-4.7-Flash (demo)"
    post_id = storage.save(result)

    print(f"\n\n{'═' * 65}")
    print("  ФИНАЛЬНЫЙ ПОСТ")
    print(f"{'═' * 65}\n")
    print(result["post"])

    print(f"\n\n{'═' * 65}")
    print("  SEO-МЕТАДАННЫЕ")
    print(f"{'═' * 65}")
    seo = result["seo"]
    print(f"\nМета-описание:\n  {seo.get('meta_description', '')}")
    print(f"\nВарианты заголовков:")
    for v in seo.get("headline_variants", []):
        print(f"  [{v['type']:8}] {v['text']}")
    print(f"\nТеги: {', '.join(seo.get('tags', []))}")
    print(f"Хэштеги: {' '.join(seo.get('hashtags', []))}")
    print(f"Время публикации: {seo.get('best_posting_time', '')}")

    print(f"\n\n{'═' * 65}")
    print("  СТАТИСТИКА PIPELINE")
    print(f"{'═' * 65}")
    stats = result["pipeline_stats"]
    step_times = stats["step_times"]
    print(f"\n  Шаги:")
    for step, t in step_times.items():
        bar = "█" * int(t * 3) if t > 0 else "·"
        print(f"    {step:10} {t:5.2f}с  {bar}")
    print(f"\n  Итого:       {stats['total_time']}с")
    print(f"  Токены ~:    {stats['token_usage']['total_tokens']}")
    print(f"  Качество:    {result['quality_score']}/10")

    qb = result.get("quality_breakdown", {})
    if qb:
        print(f"\n  Оценка редактора:")
        for criterion, score in qb.items():
            bar = "★" * score + "☆" * (10 - score)
            print(f"    {criterion:12} {score}/10  {bar}")

    print(f"\n\n{'═' * 65}")
    print("  ЭКСПОРТ В РАЗНЫЕ ФОРМАТЫ")
    print(f"{'═' * 65}")
    fmt = OutputFormatter()
    for target_fmt in ("plain", "telegram", "markdown"):
        converted = fmt.convert(
            content=result["post"],
            fmt=target_fmt,
            title=result["title"],
            tags=result["tags"],
        )
        preview = converted[:120].replace("\n", " ")
        print(f"\n  [{target_fmt:8}] {preview}...")

    print(f"\n\n{'═' * 65}")
    print("  СОХРАНЕНИЕ")
    print(f"{'═' * 65}")
    print(f"\n  post_id:  {post_id}")
    print(f"  Файлы:    {tmp_dir}/{post_id}/post.json")
    print(f"            {tmp_dir}/{post_id}/post.md")

    md_path = Path(tmp_dir) / post_id / "post.md"
    print(f"\n  Содержимое post.md (первые 10 строк):")
    lines = md_path.read_text(encoding="utf-8").splitlines()
    for line in lines[:10]:
        print(f"    {line}")

    print(f"\n{'█' * 65}")
    print("  ГОТОВО — pipeline успешно завершён")
    print(f"{'█' * 65}\n")


if __name__ == "__main__":
    main()
