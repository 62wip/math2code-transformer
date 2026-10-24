# Math2Code Transformer

Трансформер с нуля для решения текстовых математических задач через генерацию Python-кода. Репозиторий содержит реализацию модели, код обучения и инференса.

Encoder–decoder Transformer обучается на [SlavaYus/tinysms_qa_12m](https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m): текст арифметической задачи (`question`) -> Python-код с функцией `solve()`, возвращающей числовой ответ на вопрос.

Обучение — предсказание следующего токена эталонного кода (teacher forcing), cross-entropy без `<pad>`. Основная метрика — **pass@1**: доля вопросов, для которых одна выбранная программа возвращает эталонный ответ с заданным числовым допуском. Целевой лимит генерации — 180 с на всю тестовую выборку на одной NVIDIA T4.

## Структура

```text
lib/
├── models/
│   ├── baseline.py             # модель на nn.TransformerEncoder/Decoder
│   ├── baseline_plus_plus.py   # модель на собственных блоках
│   └── blocks.py               # RoPE, attention, GLU, encoder/decoder-блоки
├── data_processing/            # очистка, разбиение данных, BPE, батчи
├── experiment/                 # сборка запусков из YAML, checkpoints, resume
├── train.py                    # цикл обучения
├── evaluation.py               # разные виды валидации
├── inference.py                # загрузка весов и генерация
└── utils/                      # дополнительные функции, обертки для основных файлов
experiment_configs/             # YAML; train_example.yaml — шаблон настройки
docs/                           # документация
utilities/                      # CLI обучения, бенчмарк скорости, выгрузка в Kaggle
notebooks/
├── eda.ipynb                   # анализ данных
├── tokenization.ipynb          # обучение BPE и сохранение токенизированных данных
├── using_model_templates/      # примеры обучения, инференса и создания посылки
└── model_runs/                 # ноутбуки запусков
data/                          # локальные prepared/ и tokenized/
experiment_runs/<name>/         # config.yaml, init_state/, checkpoints/, metrics/, results/
```

`data/` и содержимое `experiment_runs/` исключены из Git.

## Модели

| `model.type` | Архитектура |
|---|---|
| `baseline_transformer` | Стандартные блоки PyTorch: MHA, FFN с ReLU, LayerNorm, обучаемые позиционные embeddings. |
| `baseline++` | Собственные pre-RMSNorm-блоки с residual и dropout: encoder — self-attention → SwiGLU; decoder — causal self-attention → cross-attention → SwiGLU. |

В `baseline++` общие embeddings вопроса и кода, RoPE в self-attention, attention через SDPA, финальные RMSNorm и линейная проекция в словарь. Опции:

- `num_kv_heads`: MHA при `null` или `num_heads`; меньшее число KV-голов включает Group Query Attention (GQA) в self/cross-attention **декодера**. Энкодер использует Multi Head Attention (MHA).
- `qk_norm`: L2-нормализация Q/K с обучаемым масштабом во всех attention-блоках.
- Число encoder/decoder-слоёв, ширина embeddings и FFN, число голов, dropout и предел длины задаются в `model.params`.

## Данные, обучение, инференс

- **Данные:** фильтрация пустых и длинных примеров, удаление комментариев и дубликатов; train/val разделены по уникальным вопросам. Общий byte-level BPE для текста и кода; `<pad>`, `<bos>`, `<eos>`, `<unk>`. Батчи группируются по длинам входа/цели с бюджетом `max_tokens`.
- **Обучение:** CPU/CUDA, DDP через `torchrun`, FP16 AMP на CUDA, gradient clipping, оптимизатор и scheduler из PyTorch, линейный warmup. Мини-эпохи задают интервал валидации, сохранения и шага scheduler. `last.pt`/`best.pt`, восстановление состояния обучения. Также есть возможность подключить логирование через W&B и лимит времени, что очень удобно для эспериментов
- **Валидация:** используются такие logit метрик: loss (CrossEntropy), token accuracy/top-5; и метрики по сгенерированному ответу модели: pass@1, BLEU, корректность синтаксиса, числовые возвраты, повторы, отсутствие EOS и таймауты.
- **Инференс:** greedy (`beam_top_k=1`) или beam search; у `baseline++` — KV-кэш self/cross-attention. `voting_mode`: `None` — лучший beam, `counter` — голосование по числовым ответам, `weighted_counter` — сумма вероятностей beams с одинаковым ответом.

## Запуск и настройки

Python ≥3.10, PyTorch ≥2.5. Из корня репозитория:

```bash
python -m pip install -e .
# Jupyter, EDA, Kaggle и W&B:
python -m pip install -e ".[notebooks,tracking]"
```

`kaggle_secrets` предоставляется средой Kaggle. Подготовка данных — `python -m lib.data_processing.data_preprocessing`; токенизация — `notebooks/tokenization.ipynb`. Обучение ожидает `data/tokenized/` с train/val и `tokenizer.json`.

Скопируй `experiment_configs/train_example.yaml`, задай уникальный `name`, пути и размеры выборок. В YAML также настраиваются модель, optimizer/scheduler, бюджет батча, число эпох, размер мини-эпохи, генерация при валидации и метрика выбора `best.pt`. Пути считаются от корня проекта; `vocab_size` и `pad_id` берутся из токенайзера.

```bash
python -m utilities.train_script --config experiment_configs/train_example.yaml --device cuda
torchrun --standalone --nproc_per_node=2 -m utilities.train_script --config experiment_configs/train_example.yaml
python -m utilities.train_script --resume experiment_runs/train_example --device cuda
```

Для CPU: `--device cpu`; для W&B: `--wandb-project <project>`; лимит времени: `--max-train-seconds <seconds>`. Каждый новый запуск требует своего `name`.

```python
from lib.inference import generate_code, load_for_inference

model, tokenizer = load_for_inference("experiment_runs/train_example", checkpoint="best")
codes = generate_code(model, tokenizer, questions=["What is 2 + 3?"], max_new_tokens=256)
```

## Итоговая модель

По итогам экспериментов выбрана **MHA 6enc / 7dec** (`mha_6enc_7dec`): вариант с GQA + QK-norm при примерно сопоставимом качестве по мини-эпохам требовал больше времени на обучение и инференс.

Архитектура: `baseline++`, 6 encoder- и 7 decoder-слоёв, embeddings 512, MHA с 8 головами, SwiGLU шириной 1536, RoPE, pre-RMSNorm, dropout 0.1, максимальная длина 1024 токена. GQA и QK-нормализация в итоговой модели отключены.

Конфигурация обучения: [mha_6enc_7dec.yaml](experiment_configs/long_runs/mha_6enc_7dec.yaml). Модель училась 16 часов.

Качество на открытой части тестовой выборки: **78.02% (730/960)**.

Проект начат в рамках курса по NLP.
