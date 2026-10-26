# q-lab

Лаборатория поиска, отсева и каталогизации торговых стратегий (crypto perp-DEX,
позже onchain DeFi). Research-слой: находит, проверяет и хранит — **но не торгует**.
Исполнение — в funding-rate-arbitrage.

Главный вопрос проекта: сколько стоит найти замену стратегии, переставшей работать.

```bash
uv sync
uv run qlab --help
```

## Консоль (только чтение)

Окно в реестр для владельца: воронка, реестр и кладбище, карточки кандидатов,
журнал прогонов. Консоль ничего не пишет — целостность реестра держится на пути
записи, второго пути в таблицы у неё нет.

Два процесса; на машине владельца они работают службами launchd (`ops/com.qlab.api.plist`,
`ops/com.qlab.console.plist`, см. `docs/STATE.md`, «Консоль»). Вручную: API (путь к БД берётся из `QLAB_DB`, по умолчанию `data/qlab.db`,
карточки — из `QLAB_CARDS`, по умолчанию `cards/`):

```bash
uv run uvicorn qlab.api:app --reload --port 8890
```

Фронт (dev-сервер проксирует `/api` на `127.0.0.1:8890`):

```bash
cd web
npm install     # один раз
npm run dev     # http://localhost:5890
```

Прод-сборка фронта — `npm run build` в `web/` (кладётся в `web/dist/`).

- План и границы — [docs/PLAN.md](docs/PLAN.md)
- Контракт реестра — [docs/REGISTRY.md](docs/REGISTRY.md)
- Задачи — [docs/TASKS.md](docs/TASKS.md)
