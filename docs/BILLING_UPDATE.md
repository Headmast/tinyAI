# Обновление тарифов биллинга - 21 марта 2026

## Изменённые тарифы

### GPT-5.4 (gpt-5.4)
| Тип | Старый | Новый |
|-----|--------|-------|
| Input | $2.50/1M | $2.50/1M (без изменений) |
| Output | $15.00/1M | $15.00/1M (без изменений) |
| Cached Input | - | $0.25/1M (добавлено) |

### GPT-5.4 Mini (gpt-5.4-mini)
| Тип | Старый | Новый |
|-----|--------|-------|
| Input | $0.10/1M | $0.75/1M |
| Output | $0.60/1M | $4.50/1M |
| Cached Input | - | $0.08/1M (добавлено) |

### GPT-5 Nano (gpt-5-nano)
| Тип | Старый | Новый |
|-----|--------|-------|
| Input | $0.05/1M | $0.20/1M |
| Output | $0.40/1M | $1.25/1M |
| Cached Input | - | $0.02/1M (добавлено) |

## Добавленные поля

В структуру моделей добавлено поле `cached_prompt_price` для поддержки кэшированных запросов:

- **gpt-5.4**: `cached_prompt_price`: 0.00025 ($0.25/1M)
- **gpt-5.4-mini**: `cached_prompt_price`: 0.00008 ($0.08/1M)
- **gpt-5-nano**: `cached_prompt_price`: 0.00002 ($0.02/1M)

## Обновленные файлы

1. **llm_cli.py** - обновлены цены в `get_available_models()`
2. **test_llm_cli.py** - обновлен тест `test_gpt54_mini_pricing`
3. **test_model_parameters.py** - обновлен тест `test_cost_calculation_accuracy`

## Текущие цены в коде (за 1K токенов)

```python
gpt-5-nano:
  prompt_price: 0.0002      # $0.20/1M
  completion_price: 0.00125 # $1.25/1M
  cached_prompt_price: 0.00002  # $0.02/1M

gpt-5.4:
  prompt_price: 0.0025       # $2.50/1M
  completion_price: 0.015    # $15.00/1M
  cached_prompt_price: 0.00025  # $0.25/1M

gpt-5.4-mini:
  prompt_price: 0.00075     # $0.75/1M
  completion_price: 0.0045  # $4.50/1M
  cached_prompt_price: 0.00008  # $0.08/1M
```

## Проверка тестов

✅ Все unit-тесты прошли успешно:
- TestModels: 6 passed
- TestModelPricing: 2 passed

## Сравнение стоимости

| Модель | Input ($/1M) | Output ($/1M) | Cached Input ($/1M) |
|--------|--------------|---------------|---------------------|
| GPT-5 Nano | $0.20 | $1.25 | $0.02 |
| GPT-5.4 Mini | $0.75 | $4.50 | $0.08 |
| GPT-5.4 | $2.50 | $15.00 | $0.25 |

**Вывод**: GPT-5.4 Mini в 3.3 раза дешевле GPT-5.4 по input и в 3.3 раза по output.
