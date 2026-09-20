# Реестр — контракт схемы

Единственный источник истины о стратегиях: что попробовали, на чём убили, каким
правилом какой версии, на каких данных. Всё остальное (досье, UI, воронка) —
представления поверх этих таблиц.

Три инварианта, которые нельзя нарушать:

1. **Вердикт — строка с числами, а не текст.** Правило, версия правил, метрика,
   значение, порог. Иначе пересмотр критериев стоит дней ручной работы.
2. **`trial` — append-only, пишутся ВСЕ прогоны**, включая выброшенные. Дефляция
   Шарпа считается по фактическому числу испытаний, а не по заявленному.
3. **У каждого вердикта есть диапазон данных.** Воскрешение с кладбища
   засчитывается только на данных, которых не было в момент отказа.

Движок: SQLite + SQLAlchemy 2.0 **синхронный** (research-инструмент, async не нужен)
+ Alembic. БД — `data/qlab.db`.

## Таблицы

### `driver` — от чего живёт эдж
Ключ к диверсификации скамейки: две стратегии на одном драйвере умрут в один день.

| поле | тип | смысл |
|---|---|---|
| `id` | str PK | slug, напр. `perp-funding-premium` |
| `title` | str | |
| `description` | str | механизм: кто и за что платит |
| `kill_condition` | str | что должно произойти на рынке, чтобы эдж исчез |
| `observable` | str | как наблюдать драйвер напрямую (метрика + источник) |

### `idea` — кандидат
| поле | тип | смысл |
|---|---|---|
| `id` | str PK | slug |
| `title` | str | |
| `source_type` | enum | `paper`/`github`/`forum`/`venue-event`/`graveyard`/`internal` |
| `source_url` | str? | |
| `claimed_edge` | str? | что утверждает источник (не проверено) |
| `asset_class` | enum | `crypto-perp`/`crypto-spot`/`defi`/`fx` |
| `driver_id` | FK? | |
| `profile` | enum | `carry`/`momentum`/`mean-reversion`/`arb`/`other` |
| `status` | enum | `candidate`→`speccing`→`implemented`→`validated`→`bench`→`live`; терминальные: `rejected`, `retired` |
| `created_at`/`updated_at` | dt | |
| `notes` | str? | |

### `spec` — формализация идеи
`id`, `idea_id` FK, `version` int, `params` json, `data_requirements` json,
`rebalance` str, `costs_model` json, `code_ref` str (модуль в `qlab.strategies`),
`created_at`. Уникально `(idea_id, version)`.

### `data_snapshot` — воспроизводимость данных
`id` str PK (sha256 содержимого), `source`, `instruments` json, `range_start`,
`range_end`, `fetched_at`, `path`, `rows`. Прогон ссылается на снапшот, а не на «API».

### `trial` — прогон (append-only)
| поле | смысл |
|---|---|
| `id` PK | |
| `spec_id` FK | |
| `config_hash` | sha256 от нормализованных params |
| `snapshot_id` FK? | null допустим только для импортированных исторических прогонов |
| `code_sha` | git sha кода стенда |
| `params` json | |
| `started_at`/`finished_at` | |
| `metrics` json | `sharpe_net`, `ann_return_net`, `max_dd`, `turnover`, ... |
| `status` | `ok`/`error` |
| `kept` bool | вошёл ли в отчёт — **на подсчёт испытаний не влияет** |
| `token_cost` int? / `cpu_seconds` float? | для метрики «цена выжившего» |
| `source` | `qlab`/`imported` — импортированные из frab помечаются явно |

### `verdict` — решение стадии, вынесенное кодом
| поле | смысл |
|---|---|
| `id` PK | |
| `idea_id` FK, `spec_id` FK?, `trial_id` FK? | |
| `stage` | `preflight`/`edge`/`tail`/`profile`/`capacity`/`correlation` |
| `rule_id`, `rules_version` | какое правило какой версии |
| `metric`, `value`, `comparator`, `threshold`, `passed` | |
| `data_range_start`, `data_range_end` | **на каких данных вынесен** |
| `decided_at`, `note` | |

### `revival_check` — дозревание кладбища
`idea_id`, `verdict_id`, `rejected_data_end` (конец данных на момент отказа),
`eligible_at` (когда накопится достаточно новых данных), `checked_at`, `outcome`
(`revived`/`still-dead`/`pending`).

### `stage_transition` — воронка
`idea_id`, `from_status`, `to_status`, `at`, `reason`, `rules_version`.
Воронка и выход по стадиям считаются отсюда, а не из текущего `status`.

### `token_spend` — бюджет
`at`, `stage`, `idea_id?`, `agent`, `tokens_in`, `tokens_out`, `usd_est`.
Нужен для главной метрики проекта — **цены одного выжившего**.

### `dossier`
`idea_id`, `path`, `rendered_at`, `rules_version`.

## Правила отбора — версионируемый конфиг

`rules/<version>.yaml`, где version — `YYYY-MM-DD.N`:

```yaml
version: "2026-09-20.1"
based_on: null           # предыдущая версия, если это правка
rules:
  - id: capital_fit
    stage: preflight
    metric: min_notional_usd
    comparator: "<="
    threshold: 2500
    fatal: true          # fatal-правило прекращает обработку кандидата
    rationale: "книга $15k, не больше 1/6 в одну стратегию"
  - id: net_edge_positive
    stage: edge
    metric: ann_return_net
    comparator: ">"
    threshold: 0.0
    fatal: true
  - id: sharpe_floor
    stage: edge
    metric: sharpe_net
    comparator: ">="
    threshold: 0.8
    near_margin: 0.25    # «почти прошёл» для запросов по кладбищу
retired:
  - id: decorrelation_required
    retired_in: "2026-09-12.1"
    reason: "цель — заработок, а не диверсификация; корреляция влияет на размер, не на допуск"
```

Движок правил — **чистая функция** `evaluate(metrics: dict, rules: RuleSet) -> list[VerdictRow]`.
Никакой агент вердикт не выносит.

## Три запроса, ради которых всё строится

1. `killed_by_retired_rules()` — отказы по правилам, которых больше нет в текущей версии.
2. `near_threshold(margin)` — отказы, где `|value − threshold| ≤ margin · |threshold|`.
3. `ripe_for_revival(min_new_days)` — идеи, у которых с момента отказа накопилось
   ≥ N дней данных, не виденных при вынесении вердикта.
