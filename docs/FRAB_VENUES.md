# Где ещё можно запускать FRAB: децентрализованные биржи (2026-10-25)

Задание владельца, 2026-10-25: «resercher нужно дать задание поискать Dex где можно запускать frab».
Все ссылки открывались **2026-10-25**; цифры снимались публичными запросами в этот же день.
Это справка по текстам условий и публичным данным, а не юридическое заключение.

## Короткий итог

1. **Ни одна децентрализованная биржа не лучше Hyperliquid для FRAB.** FRAB держит монету (спот) и
   против неё короткий бессрочный контракт, чтобы получать плату между держателями длинных и
   коротких позиций (ставка финансирования). Нужны обе ноги, и у спота должна быть живая торговля.
   На Hyperliquid спот с оборотом больше $1 млн за сутки есть у шести монет (HYPE, BTC, ZEC, ETH,
   SOL, PUMP). Лучшие из остальных: Backpack — одна монета (BTC), Aster — одна (ETH), Pacifica —
   ни одной (единственный спот, SOL, торгуется на $0,1 млн в сутки).
2. **Ставка финансирования у других площадок не выше, чем на Hyperliquid.** Среднее по семи общим
   монетам за 90 дней: Hyperliquid 9,4% годовых, Pacifica 9,0%, Extended 8,0%, Backpack 7,9%,
   Aster 6,6%; **dYdX 2,6%** (у неё нет «процентной» части формулы, на BTC и ETH ставка около нуля).
   За 365 дней все ниже: Hyperliquid 6,0%, Pacifica 5,5%, Extended 5,6%, Backpack 4,8%, Aster 4,1%.
3. **Что другие площадки дают на самом деле:** запасной вариант на случай сбоя Hyperliquid, вторую
   цену на одни и те же монеты (можно ставить короткую ногу там, где ставка выше, а спот держать
   на Hyperliquid или Binance) и данные для проверки на стенде. Это ценно, но это не «больше
   монет со спотом».
4. **Закрыты (Беларусь названа в условиях):** Lighter, Nado, Bluefin и — новое, раньше не
   проверялась — **GRVT** (страница «Restricted Jurisdictions» от 2026-08-03). Drift и Vertex не
   работают.
5. **Первыми в данные стенда (раздел «Рекомендация»):** Pacifica, Extended, Backpack. Отдельный
   дешёвый шаг: подключить спот Aster, перпы Aster на стенде уже есть.

## Пояснения терминов

- **FRAB** — купить монету за доллары и одновременно продать (в шорт) бессрочный контракт на
  эту же монету на ту же сумму. Цена монеты гасится между ногами, остаётся ставка финансирования
  (`docs/FRAB_LIVE.md`).
- **Бессрочный контракт (перп)** — контракт без даты исполнения, цена привязана к цене монеты
  регулярными платежами между длинными и короткими позициями; частота платежей — «интервал».
- **Ставка финансирования** — размер этого платежа. Если она положительная, платят длинные,
  получают короткие — то есть FRAB зарабатывает.
- **Годовая цифра** — как посчитано (одинаково для всех площадок): каждый платёж — доля суммы
  позиции за свой интервал в `h` часов; средняя ставка в час = сумма платежей / сумма часов; год =
  ×8760. Окно отсчитывается назад от 2026-10-25. Скрипт: `scripts/sample_venue_funding.py`,
  сырые данные: `docs/FRAB_VENUES_funding_2026-10-25.csv`.
- **Круг комиссий** — четыре сделки: вход в обе ноги и выход из обеих. «Берущая заявка» (taker) —
  сделка по рынку, «выставляющая заявка» (maker) — заявка, которая ждёт в книге. Считаю по
  базовому уровню комиссий, без скидок за объём и за жетоны: при капитале до $15k уровней не
  достичь.
- **Единая маржа** — один залог на спот и контракты; важно, чтобы купленная монета могла служить
  залогом для короткого контракта.
- **KYC** — проверка личности по паспорту.
- **Книга заявок против RFQ** — в книге заявок цену задают все; в RFQ (Variational) один
  маркет-мейкер называет цену по запросу.

## Как отбирались площадки

Правило владельца (`docs/SEARCH_CONSTRAINTS.md`, пункт 2): площадка закрыта, **только если её
условия называют Беларусь прямо**; общая оговорка про санкции без списка стран не закрывает.
Для каждой площадки я читал условия целиком (скриптом) и искал слово Belarus; цитата ниже —
дословно, английский оригинал. Где страница — приложение на JavaScript и текст не отдаётся
запросом, это сказано отдельно.

Обзор в `docs/VENUES.md` (разделы 1–8) использован как отправная точка; проверка «Беларусь названа
или нет» **повторена мной** для Hyperliquid, Aster, dYdX, Paradex, Orderly, Apex, edgeX, Extended,
Variational, Nado, Jupiter, Lighter, Bluefin и Backpack. Не повторена для GMX (страница не отдаёт
текст без браузера, остаётся цитата из `VENUES.md`, 3.5) и Drift (условия не получаются). Прежний вывод подтвердился везде, кроме двух
уточнений: у Backpack прочитано само пользовательское соглашение (раньше не читалось), у GRVT
найдена страница со списком стран (раньше «не проверено»).

## Таблица 1. Рейтинг по пригодности для FRAB: доступ, ноги, спот, ликвидность

Ранг: сначала площадки, где есть обе ноги и живой спот, затем — где только контракт (спот берётся
на Hyperliquid или Binance, его правило доступа проверено в `VENUES.md`, раздел 8), затем —
структурно не подходящие. Объём — за последние 24 часа на момент 2026-10-25 около 17:00 UTC,
сумма по контрактам самой площадки (способ указан в разделах ниже). «Спот ≥ $1 млн» — число
монет, у которых есть и контракт, и спот-пара с таким оборотом за сутки.

| # | Площадка | Доступ | Обе ноги тут? | Спот-пар / из них с контрактом / с оборотом ≥ $1 млн | Объём контрактов / открытые позиции | Заметка |
|---|---|---|---|---|---|---|
| 1 | **Hyperliquid** | ✓ (Беларусь не названа; общая оговорка про санкции) | да, единый счёт | 347 / 19 монет / **6** | $2,79 млрд / $11,1 млрд | основа сегодняшнего FRAB; спот у остальных 159 контрактов нет |
| 2 | **Backpack** | ✓ (Беларусь не названа; KYC обязателен) | да, единая маржа | 88 / 50 / **1** (BTC; ещё SOL $0,98 млн, ETH $0,35 млн, HYPE $0,18 млн) | $51,7 млн / $114 млн | единственная вторая площадка с несколькими монетами «спот + контракт» |
| 3 | **Pacifica** | ✓ (Беларусь не названа*; KYC не нашёл) | да, но спот один (SOL) | 1 / 1 / **0** | $400,7 млн / $123 млн (DefiLlama: $61,4 млн) | залог спотом с «бонусом за хедж» прописан в документации; глубина контрактов самая большая среди новых |
| 4 | **Aster** | ✓ (Беларусь не названа) | да, но спот тонкий | 77 / 46 / **1** (ETH $1,13 млн; BTC $0,20 млн) | $731 млн / $1,42 млрд (DefiLlama) | на стенде уже есть, спота нет |
| 5 | **Extended** | ✓ (Беларусь не названа) | нет (3 пары спота без торгов) | 3 / 0 / 0 | $183 млн / $205 млн (DefiLlama: $98,9 млн) | дешёвые комиссии, 223 крипто-контракта |
| 6 | **edgeX** | ✓ (Беларусь не названа) | нет | 0 | $474 млн / $131 млн (DefiLlama) | история ставок всего ~5 месяцев; интервал 4 часа |
| 7 | **Apex Omni** | ✓ (Беларусь не названа) | нет | 0 | $369 млн / $75 млн (DefiLlama) | на тонких монетах ставка «приклеена» к нижней границе формулы |
| 8 | **Variational** | ✓ (Беларусь не названа) | нет | 0 | $637 млн / $948 млн (DefiLlama) | **нет ни истории, ни API для ордеров** — только справочная выдача |
| 9 | **Paradex** | ✓ (Беларусь не названа) | нет (спот ETH без торгов) | 2 / 1 / 0 | $10,3 млн / $10,5 млн | объём упал: BTC $1,4 млн за сутки |
| 10 | **dYdX v4** | ✓ (Беларусь не названа) | нет | 0 | $33,5 млн / $50,4 млн | ставка на BTC/ETH около нуля — для короткой ноги бесполезна |
| 11 | **Orderly** | ✓ для ядра (у интерфейсов-брокеров свои условия) | нет | 0 | $39,8 млн / $45 млн | |
| 12 | **Aevo** | не проверено (текст условий не получен) | нет | 0 | BTC: $3,1 млн покупок за сутки / $11,1 млн (DefiLlama) | рынок крошечный |
| 13 | GMX v2 | ✓ (общая оговорка про эмбарго) | нет, пулы | — | ОИ: $58,7 млн + $213 млн (Solana) | структурно не подходит (см. ниже) |
| 14 | Jupiter Perps | ✓ (общая оговорка) | нет | — | ОИ $45 млн | структурно не подходит; API для ордеров «в работе» |
| — | Lighter | **✗** Беларусь названа | | | | |
| — | Nado (преемник Vertex) | **✗** Беларусь названа | | | | |
| — | Bluefin | **✗** Беларусь названа | | | | |
| — | **GRVT** | **✗** Беларусь названа (новое) | | | | |
| — | Drift | не работает | | | | взлом 2026-04-01 |
| — | Vertex | свёрнут | | | | июль 2025 |

\* Pacifica: текст условий (приложение на JavaScript) удалось получить только через сервис,
который пересказывает страницу и выдаёт цитаты кусками; их содержание — «Restricted Territories»
без Беларуси и общий пункт про «comprehensive economic sanctions». **Перед открытием счёта
прочитать страницу глазами** (https://app.pacifica.fi/term, «Last updated on June 11, 2026»).

## Таблица 2. Деньги и интерфейсы

«Ставка 7 монет» — среднее арифметическое по BTC, ETH, SOL, HYPE, XRP, DOGE, LINK, годовых %,
первая цифра за 90 дней (с 2026-07-27), вторая — за 365 дней. У edgeX история только около 150
дней, поэтому обе цифры за один и тот же короткий отрезок; у Aevo 365 дней только у BTC, ETH, SOL.
Комиссии круга — четыре сделки, база без скидок: сначала все четыре берущие (taker), затем все
четыре выставляющие (maker). Если спота на площадке нет, взят спот Binance 0,100% (maker и taker;
`docs/FRAB_BINANCE_SPOT.md`, строка «Комиссия покупки монеты на Binance»).

| Площадка | Ставка 7 монет, 90 д / 365 д | Интервал платежей | Комиссия контракта taker / maker | Комиссия спота taker / maker | Круг: taker / maker | Данные (история) | Исполнение |
|---|---|---|---|---|---|---|---|
| Hyperliquid | 9,4 / 6,0 | 1 час | 0,045% / 0,015% | 0,070% / 0,040% | **0,230% / 0,110%** | свечи: последние 5000 на запрос; ставки: полная, REST без ключа | Python SDK, ключ-кошелёк |
| Backpack | 7,9 / 4,8 | 1 час | 0,050% / 0,020% | 0,100% / 0,080% | **0,300% / 0,200%** | ставки с 2025-01-14, свечи спота с 2023-11, контрактов с 2025-01-05; без ключа | REST, подпись ED25519; Rust-клиент |
| Pacifica | 9,0 / 5,5 | 1 час | 0,040% / 0,015% | та же сетка, что у контрактов | **0,160% / 0,060%** | ставки 365 д (4000 за запрос, курсор), свечи до 4000 за запрос; без ключа | Python SDK, подпись Ed25519 |
| Aster | 6,6 / 4,1 | 1/2/4/8 ч по символам | 0,040% / 0% | 0,040% / 0,005% | **0,160% / 0,010%** | ставки с 2021-08, свечи с 2021-09; без ключа | REST: V3 — подпись «кошелька-агента», V1 с ключом закрыт для новых с 2026-03-25 |
| Extended | 8,0 / 5,6 | 1 час | 0,025% / 0% | (Binance) 0,100% / 0,100% | **0,250% / 0,200%** | ставки 365+ д (1000 за запрос), свечи; закрытые рынки отдаются; без ключа | Python SDK (`x10xchange/python_sdk`) |
| edgeX | 8,1 / 8,5 (≈150 д) | 4 часа | 0,045% / 0,040% (поля API, VIP-страницу не читал) | (Binance) 0,100% | **0,290% / 0,280%** | ставки только с ≈2026-05 | Python SDK (`edgex-Tech/edgex-python-sdk`) |
| Apex Omni | 7,5 / 6,5 | 1 час | 0,050% / 0,020% | (Binance) 0,100% | **0,300% / 0,240%** | ставки 365 д (окнами по 360 записей); без ключа | Python SDK (`ApeX-Protocol/apexpro-openapi`) |
| Variational | снимок: BTC 10,95%, HYPE 10,95%, ETH 9,9%, SOL 1,9%, LINK −2,0% | 1–8 ч, сейчас 8 | 0 (доход — разница цен покупки и продажи: BTC 1,2 б.п., ETH 1,3 б.п.) | (Binance) 0,100% | ≈0,21% (при размере до $1k) | **истории нет** | **нет API ордеров** (только чтение) |
| Paradex | снимок: BTC 9,6%, ETH 6,9%, SOL −0,2% | 8 часов (непрерывно) | розница 0% / 0%; Pro 0,045% / 0,003% | розница 0% | 0% (розница) | история не снималась (мгновенные значения каждые ~5 сек) | Python SDK (`tradeparadex/paradex-py`, обновлён 2026-10-24) |
| dYdX v4 | **2,6 / 0,5** | 1 час | 0,050% / 0,010% | (Binance) 0,100% | **0,300% / 0,220%** | ставки с 2023-10; закрытые рынки видны | собственная сеть, клиент на TypeScript |
| Orderly | 5,0 / 1,4 | 8 часов | у каждого брокера свои | | не определено | ставки 8 ч, 360 д | |
| Aevo | 7,8 / 6,4 (120 д) | 1 час | 0,080% / 0,050% | (Binance) 0,100% | **0,360% / 0,300%** | ставки по 50 за запрос, медленно | SDK не обновлялся с 2024-03 |

## Таблица 3. Годовая ставка финансирования по монетам, % (положительное — короткая нога получает)

90 дней, 2026-07-27 … 2026-10-25:

| Площадка | BTC | ETH | SOL | HYPE | XRP | DOGE | LINK | AVAX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hyperliquid | 8.44 | 8.63 | 7.14 | 9.86 | 9.53 | 11.47 | 10.74 | 5.67 |
| Pacifica | 9.15 | 8.49 | 6.87 | 9.48 | 9.04 | 10.97 | 8.65 | 10.14 |
| Extended | 8.10 | 7.22 | 6.42 | 8.80 | 7.81 | 10.83 | 6.55 | 4.13 |
| Backpack | 8.68 | 9.05 | 7.32 | 6.81 | 9.33 | 6.19 | 7.84 | 2.15 |
| Aster | 5.92 | 5.85 | 4.17 | 6.48 | 6.03 | 6.77 | 10.90 | 8.15 |
| edgeX | 4.41 | 8.12 | 8.26 | 7.08 | 9.34 | 10.12 | 9.31 | — |
| Apex Omni | 5.56 | 3.32 | −2.21 | 9.05 | 15.42 | 10.68 | 10.79 | 10.76 |
| Orderly | 8.10 | 13.09 | 0.08 | 0.90 | 4.86 | 2.92 | 5.24 | 7.38 |
| Aevo | 5.75 | 4.62 | 3.48 | 10.12 | 9.83 | 10.42 | 10.30 | 10.00 |
| dYdX v4 | 0.53 | −0.26 | −2.27 | 3.46 | −3.10 | 1.93 | 17.65 | −0.00 |
| GRVT (закрыта) | 8.70 | 6.18 | 6.55 | 4.57 | 8.67 | 10.65 | 3.80 | 8.25 |

365 дней, 2025-10-25 … 2026-10-25 (edgeX — ≈150 дней, Aevo HYPE…AVAX — 120 дней):

| Площадка | BTC | ETH | SOL | HYPE | XRP | DOGE | LINK | AVAX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hyperliquid | 5.82 | 6.40 | 0.85 | 9.70 | 3.28 | 6.02 | 9.70 | 3.51 |
| Pacifica | 5.72 | 4.40 | 0.64 | 3.64 | 5.26 | 9.96 | 9.17 | 9.15 |
| Extended | 6.62 | 5.94 | 1.53 | 7.69 | 3.21 | 6.74 | 7.24 | 3.05 |
| Backpack | 5.38 | 4.70 | 3.24 | 3.49 | 2.77 | 5.25 | 8.63 | −0.64 |
| Aster | 3.90 | 4.22 | 0.26 | 3.21 | 2.25 | 4.69 | 10.19 | 7.67 |
| edgeX | 5.33 | 8.23 | 9.13 | 6.28 | 9.93 | 10.42 | 9.89 | — |
| Apex Omni | 4.27 | 7.42 | −3.19 | 5.33 | 10.23 | 10.72 | 10.70 | 9.93 |
| Orderly | 4.28 | 9.81 | −7.62 | 1.78 | 0.62 | 1.21 | −0.25 | −1.08 |
| Aevo | 3.24 | 2.17 | −0.87 | 10.06 | 9.65 | 10.32 | 10.32 | 9.86 |
| dYdX v4 | −2.11 | −1.50 | −5.29 | 7.55 | −3.54 | 1.26 | 6.90 | 0.25 |
| GRVT (закрыта) | 5.68 | 6.00 | 2.53 | 3.10 | 3.19 | 7.05 | 5.06 | 2.31 |

Как читать эти цифры честно:

- **Почти везде около 10,95% годовых на тонких монетах (DOGE, LINK, AVAX на Apex, Aevo, edgeX) — не
  доход, а нижняя граница формулы.** Все эти площадки считают ставку как «премия плюс поправка к
  1 базисному пункту за 8 часов»; когда книга пустая и премии нет, остаётся эта константа
  (0,01% за 8 часов = 10,95% годовых). Реально получить её можно, только если сделки по такой цене
  исполняются, а на этих монетах оборот за сутки — десятки тысяч долларов. Доля часов с отрицательной
  ставкой у них близка к нулю по той же причине.
- **У dYdX нет этой константы:** по документации процентная часть для рынков с общим залогом равна
  нулю («0% for cross markets»), поэтому на BTC/ETH ставка колеблется около нуля, а на половине
  часов отрицательна (43,9% часов у ETH за 90 дней).
- **Окно 90 дней выше окна 365 дней** почти везде: рынок остыл в 2025-26 и оживает в последние месяцы.
  Выбор окна меняет вывод сильнее, чем выбор площадки.
- Срок жизни площадок разный: сравнение «за 365 дней» у молодых площадок включает период их роста.
- Доля отрицательных часов за 90 дней (BTC): Hyperliquid 8,6%, Pacifica 6,0%, Extended 10,6%,
  Backpack 7,6%, Aster 8,1%, dYdX 24,4%. Полностью — в CSV.

## Площадки по одной

### 1. Hyperliquid (основа)

- **Доступ.** Условия на сайте приложения (`app.hyperliquid.xyz/terms`, «Last updated on June 15,
  2026»; текст взят из программного пакета страницы, сама страница — приложение на JavaScript). Беларусь не названа:
  > «1.6 The Interface is not available to “Restricted Persons.” For the purposes of these Terms, Restricted Persons include: (a) persons or entities who reside in, are located in, are incorporated in, or have a registered office in the United States of America or Ontario, Canada; (b) persons or entities who reside in, are located in, are incorporated in, or have a registered office in jurisdictions subject to applicable economic and trade sanctions or export control laws and regulations (collectively, “Restricted Territories”); and (c) citizens of Restricted Territories, regardless of their location.»

  По правилу владельца — доступна (это уже оговорено в `VENUES.md`, 3.1). KYC нет.
- **Контракты.** 234 рынка в основном списке, из них 56 помечены `isDelisted` (метод `meta`,
  2026-10-25) и 178 живых. Ставка: раз в час, одна восьмая от 8-часовой формулы «премия + поправка
  (процент 0,01% за 8 часов минус премия, зажатая в ±0,05%)»
  (https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding). Плечо BTC до 40 (метод `meta`).
  Комиссии: https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees (контракт 0,045/0,015,
  спот 0,070/0,040, базовый уровень).
- **Спот.** 347 пар, 330 к USDC, но только 19 монет имеют и спот-пару, и контракт (способ: имя
  токена спота равно имени контракта или «U» + имя, как UBTC → BTC; `spotMetaAndAssetCtxs` +
  `metaAndAssetCtxs`). Суточный оборот спота по этим монетам: HYPE $20,8 млн, BTC $12,6 млн, ZEC
  $9,4 млн, ETH $6,9 млн, SOL $2,3 млн, PUMP $1,2 млн, PURR $0,42 млн, ENA $0,16 млн; у остальных
  — меньше $40 тыс. Весь спот Hyperliquid — $62,8 млн в сутки против $2,79 млрд контрактов.
- **Маржа.** Единый счёт: спот-баланс служит залогом для кросс-позиций
  (https://hyperliquid.gitbook.io/hyperliquid-docs/trading/account-abstraction-modes); есть портфельная
  маржа для HYPE, BTC, USDC, USDT.
- **Данные.** `POST https://api.hyperliquid.xyz/info`: `fundingHistory` (полная), `candleSnapshot`
  (последние 5000 свечей), закрытые рынки отдаются по имени. Лимит: 1200 «веса» в минуту на IP
  (https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/rate-limits-and-user-limits).
- **Исполнение.** Python SDK (`hyperliquid-dex/hyperliquid-python-sdk`, обновлён 2026-06-04),
  подпись ордеров ключом кошелька или «кошелька-агента»; газа за сделки нет.

### 2. Backpack

- **Доступ.** Прочитано полностью пользовательское соглашение (PDF, «Last Updated: September 7,
  2026», ссылка с https://support.backpack.exchange/legal/general-legal/user-agreement.md);
  слова Belarus в нём нет. Определение запрещённого места, п. 7.1.1:
  > «you are not resident, located in or otherwise attempting to access the Platform or the Services from a Restricted Location. For the purposes hereof, “Restricted Location” means any location where Backpack Exchange is not made available including without limitation any sanctioned jurisdiction and such other locations as specified by Backpack Exchange from time to time»

  Справочная страница «Supported Regions» (https://support.backpack.exchange/exchange/exchange-account/identity-verification/supported-regions.md):
  > «We currently do not serve residents of the following countries and regions:»

  — далее Северная Корея, Иран, Куба, Крым, Севастополь, Луганская и Донецкая области; «Canada, Japan, New Zealand» — «not currently able to serve». Беларуси нет.
  Приложение A соглашения в строке «All other services» для «Residents of all other jurisdictions»
  называет исполнителем «3-102-939537 SOCIEDAD DE RESPONSABILIDAD LIMITADA» (Коста-Рика). **KYC обязателен** — страна в паспорте
  будет видна. Оговорка: «sanctioned jurisdiction» без списка — по правилу владельца не закрывает.
- **Контракты и спот.** 103 бессрочных, 88 спот-пар (`/api/v1/markets`); пересечение — 50 монет.
  Оборот за сутки: BTC спот $2,16 млн / контракт $22,5 млн; SOL $0,98 / $5,6 млн; ETH $0,35 / $7,5
  млн; HYPE $0,18 / $2,3 млн; у остальных — меньше $50 тыс. Весь спот $5,0 млн, контракты $51,7
  млн. Закрытых контрактов 11 (история ставок у них отдаётся).
- **Ставка.** Каждый час (https://support.backpack.exchange/exchange/trading/futures/fees.md; «most
  pairs settling hourly»). Расчёты в USDC.
- **Маржа.** Единая кросс-маржа: спот-балансы — залог с понижающим коэффициентом
  (https://support.backpack.exchange/exchange/trading/margin-system.md; «Unified cross-margin. Your
  eligible assets (spot balances, perps margin, lending balances) live in one collateral pool»).
- **Комиссии.** https://support.backpack.exchange/exchange/trading/trading-fees.md: Tier 1 спот
  0,08%/0,10% (maker/taker), контракты 0,02%/0,05%.
- **Данные.** `GET https://api.backpack.exchange/api/v1/fundingRates?symbol=…&limit=1000&offset=…`
  (BTC с 2025-01-14), `klines` (спот BTC_USDC с 2024-04-01, SOL_USDC с 2023-11-18, контракт BTC с
  2025-01-05), без ключа.
- **Исполнение.** Подпись ED25519 (https://docs.backpack.exchange/), Rust-клиент
  `backpack-exchange/bpx-api-client` (обновлён 2026-10-20).

### 3. Pacifica

- **Доступ.** Беларусь не названа (см. сноску к таблице 1). По пересказу: «not available to
  persons or entities who reside in, are located in» США, Канады, Великобритании, Китая, России,
  Украины, Кубы, Ирана, Венесуэлы, Сирии, Северной Кореи, Филиппин «or any other jurisdiction
  subject to comprehensive economic sanctions». KYC в проверенных страницах не упомянут.
- **Контракты и спот.** 77 бессрочных и 1 спот (SOL-USDC), метод `GET https://api.pacifica.fi/api/v1/info`;
  оборот контрактов $400,7 млн, BTC $210 млн, ETH $137 млн, SOL $34 млн, HYPE $6,8 млн; прочие
  монеты (XRP, DOGE, LINK, AVAX) — до $0,12 млн. Спот SOL — $0,10 млн в сутки; это единственная
  монета в `GET /api/v1/spot_assets` (залог под 80% стоимости).
- **Ставка.** Раз в час, формула как у Hyperliquid: «funding_rate = (premium_index + clamp(interest_rate
  - premium_index, -0.05%, 0.05%))/8», интерес 0,01% за 8 часов
  (https://pacifica.gitbook.io/docs/trading-on-pacifica/funding-rates.md).
- **Маржа.** «Unified margin is a generalization of cross margin across spot and per assets»; спот
  входит в залог с коэффициентом LTV, а часть, «хеджированная» коротким контрактом на ту же монету
  в кросс-режиме, получает надбавку (документация «Spot Collateral»: «may be boosted for the portion
  of the balance hedged by an opposing cross-margin short»). Предел залога на одну монету — $10 000
  по умолчанию.
- **Комиссии.** https://pacifica.gitbook.io/docs/trading-on-pacifica/trading-fees.md: уровень 1 —
  maker 0,015%, taker 0,040%; сетка спота та же («Fee tiers follow the perpetual fee schedule»).
- **Данные.** `GET /api/v1/funding_rate/history?symbol=BTC&limit=4000&cursor=…` (365 дней вышло за 3
  запроса), свечи `kline` до 4000 за запрос; без ключа. Закрытые рынки в `info` не видны —
  честный список монет на момент прошлых дат отсюда не получить.
- **Исполнение.** Python SDK `pacifica-fi/python-sdk` (обновлён 2026-02-16), подпись Ed25519.
  Площадка молодая: контракты созданы 2025-06-02, спот — 2026-04-19.

### 4. Aster

- **Доступ.** Условия от 2026-02-25, п. 2.1, список содержит Россию и Украину, но не Беларусь:
  > «the United States of America, Canada, the United Kingdom, China, North Korea, Russia, Ukraine, Cuba, Iran, Venezuela, or Syria. This prohibition extends to any jurisdiction or territory that is subject to comprehensive country-wide or regional economic sanctions»

  Проверка личности: «limited identity verification procedures» (`VENUES.md`, 3.2).
- **Контракты и спот.** 605 контрактов в USDT (и 32 в других котируемых активах); спот 77 пар
  (`https://sapi.asterdex.com/api/v1/exchangeInfo`), 73 монеты, пересечение с контрактами — 46.
  Оборот: контракты $731 млн, спот всего $19,7 млн; спот ETH $1,13 млн, ASTER $0,78 млн, BTC $0,20
  млн, SOL $0,02 млн. Ставка: интервал 1/2/4/8 часов по символам (BTC, ETH, SOL — 8, HYPE — 4).
- **Комиссии.** https://docs.asterdex.com/trading/perpetuals/fees-and-specs/fees.md:
  контракты 0% / 0,04%; спот 0,005% / 0,04%.
- **Маржа.** Режим «Multi-Asset Mode»: залог — разные монеты, общий на все позиции
  (https://docs.asterdex.com/trading/perpetuals/single-asset-mode-and-multi-asset-mode.md).
- **Данные и исполнение.** Подробности — `VENUES.md`, 3.2 и раздел 7. Исполнение: «V3» —
  подпись кошелька-агента (nonce, `/fapi/v3/*`); старые API-ключи (`V1`) новыми не создаются с
  2026-03-25 (https://github.com/asterdex/api-docs).

### 5. Extended

- **Доступ.** Страница «Restricted Countries»:
  > «“Restricted Persons” includes any individual or entity residing in, located in, incorporated in, or with a registered office in any jurisdiction in which access to or use of the Extended Protocol is restricted, including without limitation: (A) Afghanistan, Canada, China (Hong Kong), Cuba, Iran, Iraq, Yemen, North Korea, Russia, Syria, Seychelles, Ukraine (Crimea, Donetsk, Kherson, Luhansk and Zaporizhzhia), the United Kingdom and the United States of America; and (B) any other individual or entity whose access to or use of the Extended Protocol, website, API, and services would be contrary to any applicable law or regulatory requirement.»

  Беларусь не названа (и во всём тексте «Terms of Use» слова нет). Охват включает API.
- **Контракты.** 407 рынков (`/api/v1/info/markets`): 223 крипто, 181 традиционные активы
  («RWA»), 3 пары спота без торгов. Статусы: ACTIVE 333, DELISTED 62 (отдаются вместе с историей
  ставок), PRELISTED 8. Оборот $183,3 млн; BTC $71,9 млн, ETH $20,9 млн, HYPE $9,9 млн, SOL $3,0
  млн, XRP $1,5 млн, остальные из восьми — менее $0,15 млн. Плечо BTC до 50, минимальный ордер
  BTC 0,0001.
- **Ставка.** Раз в час, формула как у Hyperliquid (https://docs.extended.exchange/extended-resources/trading/funding-payments.md).
- **Комиссии.** «Currently, Extended features a flat fee structure for both perpetuals and spot
  markets: Taker: 0.025% Maker: 0.000%»
  (https://docs.extended.exchange/extended-resources/trading/trading-fees-and-rebates.md).
- **Маржа.** Единая маржа; залог USDC, wBTC и ETH с коэффициентом 90%
  (https://docs.extended.exchange/extended-resources/trading/unified-margin-and-balances.md) —
  но нет живого спота на самой площадке.
- **Данные.** `GET /api/v1/info/{market}/funding` (1000 записей на запрос, 365 дней выгружены),
  свечи часовые. **Исполнение.** Python SDK `x10xchange/python_sdk` (обновлён 2026-09-04).

### 6. edgeX, Apex Omni

- **edgeX.** Условия: «you must not be a resident, citizen or agent of, or incorporated in, and
  do not have a registered office in Iran, Cuba, North Korea, Syria, Myanmar (Burma), the regions
  of Crimea, Donetsk or Luhansk, or any other country or region that is the subject of comprehensive
  country-wide or region-wide economic sanctions by the United States» — Беларусь не названа.
  191 контракт (90 — крипто и сырьё без акций и валют: 98 акций, 3 пары валют); интервал 4 часа,
  история ставок только с ≈2026-05 (старые контракты в USDT пустые). Комиссии из метаданных BTC:
  `defaultMakerFeeRate` 0,0004, `defaultTakerFeeRate` 0,00045; плечо BTC 100. Python SDK
  `edgex-Tech/edgex-python-sdk` (2026-09-10).
- **Apex Omni.** Условия — тот же шаблон, что у dYdX: «… or any other country or region that is the
  subject of comprehensive country-wide or region-wide economic sanctions by the United States» —
  Беларусь не названа. 139 контрактов (у 50 торговля выключена, список закрытых виден), только кросс-маржа в USDT, залог ETH/WBTC и другие
  (https://apex-pro.gitbook.io/apex-pro/apex-omni/margin-risk-management-and-liquidations.md);
  комиссии уровня 1: 0,020% / 0,050% (https://apex-pro.gitbook.io/apex-pro/apex-omni/trading-and-funding-fees.md).
  Интерфейс истории ставок отдаёт не более 360 записей за окно (15 дней), глубже — сдвигом конца
  окна. Python SDK `ApeX-Protocol/apexpro-openapi` (2026-09-21).

### 7. Variational, Paradex, dYdX v4, Orderly, Aevo

- **Variational.** «Restricted Persons» (обновлено 2026-04-20) и «Terms of Use» (2026-09-11) —
  слова Belarus нет; список территорий: Крым, Донецк, Луганск, Севастополь, Куба, Иран, Северная
  Корея, Судан, Сирия, Канада, США, территории США, Тайвань. Формула, общая для остальных
  (премия + поправка), окно расчёта 1–8 часов; торговых комиссий нет («There are no trading fees on
  Omni»), доход площадки — разница цен покупки и продажи. 582 рынка (564 бессрочных), BTC оборот
  $306 млн, разница цен 1,17 б.п. Публичный API — только `GET /metadata/v2/stats` (текущая
  сводка, лимит 10 запросов за 10 секунд на IP); документации на размещение ордеров нет →
  **для FRAB пока неисполнима и без истории для стенда.**
- **Paradex.** Страница «Restricted Persons»: общая формулировка про санкции США/ЕС/Великобритании/
  Сингапура/ООН плюс перечень, где Беларуси нет (`VENUES.md`, 3.7: «условия противоречат сами
  себе»). Есть спот (2 пары) и опционы, но объёмы: контракты $10,3 млн (66 рынков), спот ETH — ноль,
  BTC за сутки $1,4 млн. Ставка — взвешенная медиана премий по Paradex, Binance, Bybit, OKX,
  Hyperliquid и Lighter (https://docs.paradex.trade/risk/funding-mechanism.md), то есть
  повторяет другие площадки. Розничные комиссии нулевые
  (https://docs.paradex.trade/trading/trading-fees.md).
- **dYdX v4.** Условия от 2026-03-12: «“Restricted Territory” means Iran, Cuba, North Korea, the
  regions of Crimea, Donetsk or Luhansk, or any other country or region that is the subject of
  comprehensive country-wide or region-wide economic sanctions by the United States» — Беларусь
  не названа. Ставка раз в час, процентной части для рынков с общим залогом нет
  (https://docs.dydx.xyz/concepts/trading/funding). Комиссии базовые: берущая 5,0 б.п., выставляющая 1,0 б.п.
  (https://docs.dydx.xyz/concepts/trading/rewards). Спот-пар на индексаторе сети нет (только
  `perpetualMarkets`); о споте dYdX Labs объявила как о запуске для Solana (The Block, 2025-12-11) —
  устройство не изучал. Закрытые рынки: 223 в статусе
  финального расчёта видны (`VENUES.md`, 7.1). Данные и всё прочее — `VENUES.md`, 3.4.
- **Orderly.** Условия: запрещены США и «comprehensive country-wide … sanctions by the United
  States (including, without limitation, Cuba, Democratic People's Republic of Korea, Iran)»;
  Беларусь не названа. Это инфраструктура — условия брокеров-интерфейсов не читал. Интервал 8
  часов, ставки по 500 за запрос.
- **Aevo.** Страницу условий получить не удалось (приложение на JavaScript, документация
  отсылает на `app.aevo.xyz/terms` без текста) — **доступ не проверен**. 102 контракта, плечо BTC
  до 20, минимальный ордер $10; комиссии 0,08% / 0,05% (https://www.aevo.xyz/docs/aevo-products/aevo-exchange/fees/perpetuals-fees.md).
  Рынок мелкий: ОИ $11 млн.

### 8. GMX v2 и Jupiter Perps: почему структурно не подходят

- **GMX.** Условия (`VENUES.md`, 3.5): «You are not a resident, national, or agent of any country
  to which the United States, the United Kingdom, or the European Union embargoes goods or imposes
  similar sanctions» — общая оговорка, Беларусь не названа. Цена берётся у оракула, книги заявок
  нет; ставка финансирования плавающая и зависит **только от перекоса длинных и коротких
  позиций**: платит сторона с большим интересом («Funding fees may be a cost or a credit while a
  position is open»), а сторона с большим открытым интересом ещё платит «borrow fee»
  (https://docs.gmx.io/docs/trading/fees.md). Короткая нога FRAB получает плату и не платит за заём,
  пока коротких меньше, чем длинных; когда коротких больше, она платит и ставку, и заём. Размер
  платы задаёт перекос внутри одного рынка, а не «рыночная» ставка, поэтому прямого аналога FRAB
  тут не получится. Ордера — ончейн-транзакции (Arbitrum, Avalanche, MegaETH), SDK на TypeScript
  (`@gmx-io/sdk`); простой API истории ставки не найден.
- **Jupiter Perps.** Оговорка про санкции США/Великобритании/ЕС без списка (`VENUES.md`, 3.14);
  контракты только на SOL, ETH, BTC, один общий пул. Документация: «The Perps API is still a work
  in progress» (https://developers.jup.ag/docs/perps/index.md) — ордера только прямыми вызовами
  программы. Истории нет.

### 9. Закрытые и неработающие

| Площадка | Цитата (английский оригинал) | Источник |
|---|---|---|
| Lighter | «OUR SERVICES WERE NOT DEVELOPED FOR, AND ARE NOT AVAILABLE TO PERSONS OR ENTITIES WHO RESIDE IN, ARE LOCATED IN, ARE INCORPORATED IN, OR HAVE A REGISTERED OFFICE OR PRINCIPAL PLACE OF BUSINESS IN THE UNITED STATES OF AMERICA, CANADA, THE UNITED KINGDOM, CHINA, NORTH KOREA, RUSSIA, UKRAINE, CUBA, IRAN, VENEZUELA, SUDAN, BELARUS, MYANMAR OR SYRIA.» | https://lighter.xyz/terms |
| Nado | «you do not reside in, are not located or incorporated in and do not have a registered office in (i) Belarus, Cuba, Iran, North Korea, Russia, or Ukraine, or any country or region that is the subject of comprehensive country-wide or region-wide economic sanctions by the United States or the United Kingdom» | https://www.nado.xyz/legal-docs/terms-of-use |
| Bluefin | «you are not a resident, national, or agent of, located in, incorporated or otherwise formed in, or have a registered office in, Antigua and Barbuda, Algeria, Bangladesh, Bolivia, Belarus, Burundi, …» | https://bluefin.io/terms-of-use |
| **GRVT** (новое) | «Your access to or use of the Platform and Services are banned if you are a person residing, located, incorporated or otherwise established in any of the countries/ jurisdictions listed below … ❌ Belarus» («Restricted Jurisdictions», 3 августа 2026); на неё ссылается пользовательское соглашение от 14 сентября 2026 | https://help.grvt.io/en/articles/9711621-restricted-jurisdictions |
| Drift | взлом 2026-04-01; данные `data.api.drift.trade` и `mainnet-beta.api.drift.trade` не отвечают (2026-10-25), перезапуск не подтверждён | `VENUES.md`, 3.8; Bloomberg Law, 2026-04 («plans to relaunch») |
| Vertex | свёрнут в июле 2025, преемник — Nado (закрыт) | The Block, 2025-07-08 |

## Что осталось не разобрано

Из рейтинга DefiLlama по объёму открытых позиций (`https://api.llama.fi/overview/open-interest`,
2026-10-25) не открывал условий и API: Antarctic ($325 млн), QFEX ($160 млн, по описанию —
традиционные активы), SoDEX ($151 млн), Arcus ($117 млн, сеть Robinhood), Ondo Perps ($105 млн,
акции и сырьё), Vest ($88 млн), PopDEX ($72 млн), Veranta ($69 млн), Derive ($75 млн), TxFlow, RISEx,
Phoenix, Bullbit, BULK. Для Antarctic и SoDEX поиск не нашёл ни документации, ни условий. Большинство
из них — контракты на традиционные активы, для которых у FRAB нет спота. Это следующий круг
проверки, если выбранные ниже не оправдают себя.

## Рекомендация

**Главный вывод простыми словами.** На Hyperliquid FRAB уже идёт там, где нужен спот с живой
торговлей. Другие децентрализованные биржи не дают больше монет со спотом: у всех спот тонкий
или его нет. Поэтому добавлять их надо не ради «большего числа монет», а ради трёх вещей:
сравнить ставку на тех же монетах, иметь запасную площадку и проверять стратегию на независимых
данных.

**В данные стенда первыми:**

1. **Pacifica.** Самая глубокая из новых (контракты BTC $210 млн за сутки), ставка за 90 дней в
   среднем 9,0% при 9,4% у Hyperliquid, история ставок на 365 дней одним методом без ключа,
   комиссии ниже, чем у Hyperliquid (круг из четырёх сделок 0,16% против 0,23%), единая маржа со
   спот-залогом и надбавкой за хедж — задумана именно для таких позиций. Слабое место: спот только
   SOL ($0,1 млн в сутки), молодая площадка (контракты с 2025-06), условия читал через пересказ —
   перед счётом прочитать глазами.
2. **Extended.** Самые дешёвые комиссии (контракт 0,025% / 0%), $183 млн в сутки, 223
   крипто-контракта, история ставок на год+ и **список закрытых рынков с историей** (62 рынка).
   Спота нет: короткую ногу ставить тут, спот держать на Hyperliquid или Binance — на стенде это
   вариант «межплощадочный FRAB» (`docs/FRAB_BINANCE_SPOT.md`). Ставка за 90 дней 8,0%.
3. **Backpack.** Единственная вторая площадка, где обе ноги реально на одном счёте и есть
   несколько монет (BTC, SOL, ETH, HYPE со спотом на $0,2–2,2 млн в сутки), с единой маржой и
   спот-залогом. Минусы: самые высокие комиссии (круг 0,30% taker), объём контрактов всего $52
   млн, KYC по паспорту обязателен. Данные: ставки с 2025-01, спот-свечи BTC с 2024-04.

**Дешёвый шаг сверх списка:** подключить **спот Aster** — перпы Aster на стенде уже есть
(`VENUES.md`, раздел 7), а спот-пересечение 46 монет даёт вторую площадку «обе ноги» без нового
источника; глубина спота мала (ETH $1,1 млн в сутки), для капитала до $15k хватает.

**Не тратить время:** dYdX v4 (ставка на BTC/ETH около нуля — для короткой ноги нечего
собирать), Paradex (объём упал до $1,4 млн на BTC), Orderly и Aevo (мелкие и разовые
исключения), GMX и Jupiter (устройство не совпадает с FRAB), Variational (нет ни истории, ни
API ордеров; смотреть, если появятся).

**Чего эти данные не доказывают.** Ставка за 90 дней выше, чем за 365; один и тот же вывод из
окна и из другого окна может отличаться. Цифры сняты один раз и описывают прошлое. Жёсткий отсев
по доступу — по правилу владельца; практика площадок (блокировка по IP) не проверялась.
Нужна ли проверка у юриста на выгрузку данных через API — вопрос из `VENUES.md`, раздел 5, пункт 5.

## Что не удалось проверить

1. Условия Aevo и Pacifica в оригинале (получены только пересказом или не получены).
2. Условия брокеров-интерфейсов Orderly.
3. Страница GMX не отдаёт текст запросом; цитата — из `VENUES.md`.
4. Историю ставок GMX и Jupiter: простого API нет.
5. Историю ставок Paradex: публичный метод отдаёт мгновенные значения; в таблице — снимок.
6. Ставки Variational: только снимок, истории нет.
7. Минимальный ордер и плечо для каждой площадки: приведены лишь там, где их отдаёт API
   площадки (Extended, Pacifica, Aevo, edgeX, Paradex); для Hyperliquid и Backpack в этот проход
   не перепроверялись.
8. Объём по DefiLlama платный; взят объём по API самих площадок, поэтому у разных площадок
   способ суммирования разный — указано в разделах.

## Как воспроизвести

```
python3 -I scripts/sample_venue_funding.py --cache /tmp/frab_venues_cache > funding.csv
```

Условия — страницы по адресам из разделов. ОИ — `curl 'https://api.llama.fi/overview/open-interest?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true'`.
Пользовательские соглашения Backpack и GRVT — PDF по ссылкам на их справочных страницах
(скачаны во временную папку и прочитаны `pdftotext`; в репозиторий не добавлены).
