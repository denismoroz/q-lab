# q-lab

Лаборатория поиска, отсева и каталогизации торговых стратегий (crypto perp-DEX,
позже onchain DeFi). Research-слой: находит, проверяет и хранит — **но не торгует**.
Исполнение — в funding-rate-arbitrage.

Главный вопрос проекта: сколько стоит найти замену стратегии, переставшей работать.

```bash
uv sync
uv run qlab --help
```

- План и границы — [docs/PLAN.md](docs/PLAN.md)
- Контракт реестра — [docs/REGISTRY.md](docs/REGISTRY.md)
- Задачи — [docs/TASKS.md](docs/TASKS.md)
