#!/usr/bin/env python3
"""
Скрипт для сравнения трёх моделей разной мощности:
- gpt-5-nano (слабая)
- gpt-5.4-mini (средняя)
- gpt-5.4 (сильная)

Замеряет время, токены и стоимость
"""
import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from llm_cli import get_available_models, calculate_cost

# Тестовый промпт
TEST_PROMPT = """Как сформировать собственный стиль технического блога, который ведётся ИИ-агентом, но не похожим на стандартные ответы LLM."""

# Модели для сравнения (слабая -> средняя -> сильная)
MODELS_TO_TEST = [
    {
        "id": "gpt-5-nano",
        "name": "GPT-5 Nano",
        "category": "Слабая"
    },
    {
        "id": "gpt-5.4-mini",
        "name": "GPT-5.4 Mini",
        "category": "Средняя"
    },
    {
        "id": "gpt-5.4",
        "name": "GPT-5.4",
        "category": "Сильная"
    }
]

def test_model_with_streaming(client, model_id, prompt):
    """
    Тестирует модель с замером времени от начала до конца streaming
    """
    print(f"\n{'='*70}")
    print(f"Тестирование: {model_id}")
    print(f"{'='*70}")
    
    # Параметры запроса
    params = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 120000,
        "stream": True
    }
    
    # Убираем temperature для gpt-5-nano
    if model_id != "gpt-5-nano":
        params["temperature"] = 0.7
    
    full_response = ""
    
    # Засекаем время начала
    start_time = time.time()
    
    print(f"\n⏱️  Начало запроса: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    
    try:
        stream = client.chat.completions.create(**params)
        
        first_token_time = None
        
        for chunk in stream:
            if not chunk.choices or len(chunk.choices) == 0:
                continue
            
            delta = chunk.choices[0].delta
            
            if hasattr(delta, 'content') and delta.content:
                if first_token_time is None:
                    first_token_time = time.time()
                    ttft = first_token_time - start_time
                    print(f"⚡ Первый токен получен через: {ttft:.3f}s")
                
                content = delta.content
                full_response += content
                print(content, end="", flush=True)
        
        # Засекаем время окончания
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n\n⏱️  Конец запроса: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        print(f"⏱️  Общее время: {total_time:.3f}s")
        
        # Оцениваем токены (1 токен ≈ 4 символа для английского, ≈ 2-3 для русского)
        # Используем консервативную оценку 3 символа на токен
        estimated_completion_tokens = len(full_response) // 3
        estimated_prompt_tokens = len(prompt) // 3
        total_tokens = estimated_prompt_tokens + estimated_completion_tokens
        
        # Вычисляем стоимость
        models_info = get_available_models()
        model_info = models_info[model_id]
        cost = calculate_cost(estimated_prompt_tokens, estimated_completion_tokens, model_id)
        
        result = {
            "model_id": model_id,
            "model_name": model_info["name"],
            "response": full_response,
            "response_length": len(full_response),
            "word_count": len(full_response.split()),
            "prompt_tokens": estimated_prompt_tokens,
            "completion_tokens": estimated_completion_tokens,
            "total_tokens": total_tokens,
            "total_time_seconds": total_time,
            "time_to_first_token": ttft if first_token_time else 0,
            "tokens_per_second": estimated_completion_tokens / total_time if total_time > 0 else 0,
            "cost_usd": cost,
            "prompt_price": model_info["prompt_price"],
            "completion_price": model_info["completion_price"]
        }
        
        print(f"\n📊 Статистика:")
        print(f"   Токены: {total_tokens} (prompt: {estimated_prompt_tokens}, completion: {estimated_completion_tokens})")
        print(f"   Символов: {len(full_response)}")
        print(f"   Слов: {len(full_response.split())}")
        print(f"   Стоимость: ${cost:.6f}")
        print(f"   Скорость: {result['tokens_per_second']:.2f} токенов/сек")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Проверка API ключа
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("❌ Ошибка: OPENAI_API_KEY не найден в .env")
        return
    
    print("=" * 70)
    print("СРАВНЕНИЕ МОДЕЛЕЙ РАЗНОЙ МОЩНОСТИ")
    print("=" * 70)
    print(f"\nПромпт: {TEST_PROMPT}")
    print(f"\nМодели для тестирования:")
    for model in MODELS_TO_TEST:
        print(f"  • {model['name']} ({model['category']})")
    
    # Создаём клиент
    client = OpenAI(api_key=openai_api_key, timeout=120.0)
    
    results = []
    
    # Тестируем каждую модель
    for model_config in MODELS_TO_TEST:
        result = test_model_with_streaming(client, model_config["id"], TEST_PROMPT)
        if result:
            result["category"] = model_config["category"]
            results.append(result)
        
        # Небольшая пауза между запросами
        time.sleep(2)
    
    # Сохраняем результаты в JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"model_comparison_{timestamp}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n✅ Результаты сохранены в: {results_file}")
    
    # Создаём MD файл с результатами
    create_markdown_report(results, timestamp)
    
    return results

def create_markdown_report(results, timestamp):
    """Создаёт MD файл с таблицей сравнения и выводами"""
    
    md_content = f"""# Сравнение моделей разной мощности

**Дата тестирования:** {datetime.now().strftime("%d.%m.%Y %H:%M")}

## Тестовый промпт

> {TEST_PROMPT}

## Результаты сравнения

### Таблица метрик

| Метрика | GPT-5 Nano (Слабая) | GPT-5.4 Mini (Средняя) | GPT-5.4 (Сильная) |
|---------|---------------------|------------------------|-------------------|
"""
    
    # Добавляем строки таблицы
    metrics = [
        ("Общее время (сек)", "total_time_seconds", ".3f"),
        ("Время до первого токена (сек)", "time_to_first_token", ".3f"),
        ("Prompt токенов", "prompt_tokens", "d"),
        ("Completion токенов", "completion_tokens", "d"),
        ("Всего токенов", "total_tokens", "d"),
        ("Скорость (токенов/сек)", "tokens_per_second", ".2f"),
        ("Символов в ответе", "response_length", "d"),
        ("Слов в ответе", "word_count", "d"),
        ("Стоимость (USD)", "cost_usd", ".6f"),
    ]
    
    for metric_name, metric_key, fmt in metrics:
        row = f"| {metric_name} |"
        for result in results:
            value = result.get(metric_key, 0)
            row += f" {value:{fmt}} |"
        md_content += row + "\n"
    
    # Добавляем ценообразование
    md_content += f"\n### Ценообразование (за 1K токенов)\n\n"
    md_content += "| Модель | Input (USD) | Output (USD) |\n"
    md_content += "|--------|-------------|-------------|\n"
    
    for result in results:
        md_content += f"| {result['model_name']} | ${result['prompt_price']:.6f} | ${result['completion_price']:.6f} |\n"
    
    # Добавляем полные ответы
    md_content += "\n## Полные ответы моделей\n\n"
    
    for i, result in enumerate(results, 1):
        md_content += f"### {i}. {result['model_name']} ({result['category']})\n\n"
        md_content += f"**Время ответа:** {result['total_time_seconds']:.3f}s  \n"
        md_content += f"**Токенов:** {result['total_tokens']}  \n"
        md_content += f"**Стоимость:** ${result['cost_usd']:.6f}\n\n"
        md_content += "---\n\n"
        md_content += result['response'] + "\n\n"
        md_content += "---\n\n"
    
    # Добавляем выводы
    md_content += generate_conclusions(results)
    
    # Сохраняем файл
    md_file = f"model_comparison_{timestamp}.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"📄 Markdown отчёт сохранён: {md_file}")

def generate_conclusions(results):
    """Генерирует выводы на основе результатов"""
    
    if len(results) < 3:
        return "\n## Выводы\n\nНедостаточно данных для анализа.\n"
    
    nano, mini, full = results[0], results[1], results[2]
    
    conclusions = "\n## Выводы\n\n"
    
    # Сравнение по времени
    time_diff_mini_nano = ((mini['total_time_seconds'] - nano['total_time_seconds']) / nano['total_time_seconds'] * 100)
    time_diff_full_mini = ((full['total_time_seconds'] - mini['total_time_seconds']) / mini['total_time_seconds'] * 100)
    
    conclusions += f"### ⏱️ Скорость обработки\n\n"
    conclusions += f"- **GPT-5 Nano** выдала ответ за **{nano['total_time_seconds']:.2f}s** ({nano['tokens_per_second']:.1f} токенов/сек)\n"
    conclusions += f"- **GPT-5.4 Mini** работала {'быстрее' if time_diff_mini_nano < 0 else 'медленнее'} на **{abs(time_diff_mini_nano):.1f}%** ({mini['total_time_seconds']:.2f}s, {mini['tokens_per_second']:.1f} токенов/сек)\n"
    conclusions += f"- **GPT-5.4** работала {'быстрее' if time_diff_full_mini < 0 else 'медленнее'} на **{abs(time_diff_full_mini):.1f}%** ({full['total_time_seconds']:.2f}s, {full['tokens_per_second']:.1f} токенов/сек)\n\n"
    
    # Сравнение по стоимости
    cost_ratio_mini = mini['cost_usd'] / nano['cost_usd'] if nano['cost_usd'] > 0 else 0
    cost_ratio_full = full['cost_usd'] / nano['cost_usd'] if nano['cost_usd'] > 0 else 0
    
    conclusions += f"### 💰 Стоимость\n\n"
    conclusions += f"- **GPT-5 Nano**: ${nano['cost_usd']:.6f} (базовая)\n"
    conclusions += f"- **GPT-5.4 Mini**: ${mini['cost_usd']:.6f} (в **{cost_ratio_mini:.1f}x** дороже)\n"
    conclusions += f"- **GPT-5.4**: ${full['cost_usd']:.6f} (в **{cost_ratio_full:.1f}x** дороже Nano)\n\n"
    
    # Сравнение по объёму
    conclusions += f"### 📝 Объём ответа\n\n"
    conclusions += f"- **GPT-5 Nano**: {nano['word_count']} слов, {nano['response_length']} символов\n"
    conclusions += f"- **GPT-5.4 Mini**: {mini['word_count']} слов, {mini['response_length']} символов ({(mini['word_count']/nano['word_count']*100-100):+.1f}%)\n"
    conclusions += f"- **GPT-5.4**: {full['word_count']} слов, {full['response_length']} символов ({(full['word_count']/nano['word_count']*100-100):+.1f}%)\n\n"
    
    # Общий вывод
    conclusions += f"### 🎯 Итоговый вывод\n\n"
    
    # Определяем лучшую модель по соотношению цена/качество
    best_value = min(results, key=lambda x: x['cost_usd'] / x['word_count'] if x['word_count'] > 0 else float('inf'))
    fastest = min(results, key=lambda x: x['total_time_seconds'])
    most_detailed = max(results, key=lambda x: x['word_count'])
    
    conclusions += f"**Различия между моделями:**\n\n"
    conclusions += f"1. **Качество ответов**: Более мощные модели дают {('более детальные' if most_detailed['model_id'] == 'gpt-5.4' else 'сопоставимые по объёму')} ответы. "
    conclusions += f"{most_detailed['model_name']} предоставила наиболее подробный ответ ({most_detailed['word_count']} слов).\n\n"
    
    conclusions += f"2. **Скорость**: {fastest['model_name']} показала наилучшую скорость ({fastest['total_time_seconds']:.2f}s). "
    conclusions += f"Разница во времени обработки между моделями составляет от {min(r['total_time_seconds'] for r in results):.1f}s до {max(r['total_time_seconds'] for r in results):.1f}s.\n\n"
    
    conclusions += f"3. **Стоимость**: Nano-модель **в {cost_ratio_full:.0f} раз дешевле** флагманской GPT-5.4. "
    conclusions += f"По соотношению цена/качество лидирует **{best_value['model_name']}** (${best_value['cost_usd']/best_value['word_count']:.8f} за слово).\n\n"
    
    conclusions += f"4. **Рекомендации**:\n"
    conclusions += f"   - Для **быстрых и дешёвых** запросов: {nano['model_name']}\n"
    conclusions += f"   - Для **баланса** цены и качества: {mini['model_name']}\n"
    conclusions += f"   - Для **максимального** качества: {full['model_name']}\n"
    
    return conclusions

if __name__ == "__main__":
    main()
