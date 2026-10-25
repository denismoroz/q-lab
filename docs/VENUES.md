# Какие биржи бессрочных контрактов доступны резиденту Беларуси (2026-10-25)

Задача T42. Владелец — **резидент и гражданин Беларуси**. Сегодня лаборатория работает только с
Hyperliquid; вопрос — доступны ли ему другие децентрализованные биржи бессрочных контрактов.

**Жёсткое правило** (`docs/SEARCH_CONSTRAINTS.md`, пункт 2): обходов не предлагаем — ни
VPN, ни другого резидентства, ни чужих счетов, ни «интерфейсов без блокировки по стране»,
ни довода «протокол без разрешений, значит можно». Здесь только то, что сама площадка
разрешает резиденту Беларуси по собственным условиям. Где условия неясны — так и написано,
в пользу владельца вопрос не решается.

Все ссылки открывались **2026-10-25**. Это справка по текстам условий, а не юридическое
заключение: толкование оговорок — вопрос к юристу.

Пояснения терминов при первом употреблении:

- **бессрочный контракт** — вид срочного контракта без даты исполнения; цена привязана к
  обычной («спотовой») цене актива механизмом платежей между сторонами;
- **ставка финансирования** — этот регулярный платёж между держателями длинных и коротких позиций;
  интервал показывает, как часто он начисляется (раз в час, раз в 4 часа, раз в 8 часов);
- **объём открытых позиций** — суммарная стоимость всех ещё не закрытых позиций (мера «глубины» рынка);
- **свечи** — цены за период (открытие, максимум, минимум, закрытие) и объём; на них считаются проверки стратегий;
- **KYC** — проверка личности по паспорту при регистрации;
- **санкционная оговорка** — пункт условий «нельзя тем, кто в странах/списках под санкциями ЕС, Великобритании, США или ООН».

## 1. Короткий итог для владельца

**Правило владельца (2026-10-25):** «если в условиях использования для биржи явно нет указания
что не работаем с Беларусью - пока это оставляем. Примеры - bybit». Площадка считается
доступной, пока Беларусь не названа в её условиях прямо. Общие санкционные оговорки без
списка стран («страны под санкциями ЕС/США») площадку не закрывают: как сама площадка их
применяет, из текста не видно, и додумывать это нельзя. Первая версия этого раздела
трактовала такие оговорки как запрет — это был вывод, которого текст не делает.

**Закрыты (Беларусь названа в условиях):**
- **Lighter** — Беларусь в списке стран;
- **Nado** (преемник Vertex) — Беларусь в списке; сам Vertex свёрнут в 2025 году;
- **Bluefin** — Беларусь в списке (рынок крошечный: объём за сутки около $0,6 млн).

**Доступны (Беларусь в условиях не названа):**
- **Hyperliquid** — площадка, на которой лаборатория работает сегодня (в пункте 1.6 условий
  общая оговорка про страны под санкциями и экспортным контролем, без списка стран);
- **GMX**, **Jupiter Perps** — общие санкционные оговорки без списка;
- **Aster**, **dYdX**, **Apex Omni**, **edgeX**, **Orderly** (ядро; у каждого
  интерфейса-брокера свои условия), **Extended**, **Paradex**, **Variational** — Беларуси в
  списках нет;
- **Backpack** — центральная биржа с обязательной проверкой по паспорту; на справочных
  страницах Беларусь не названа, само пользовательское соглашение (файл на Google Drive)
  прочитать не удалось — проверить при открытии счёта.

**Не работает штатно:** **Drift** — взлом 2026-04-01 (потеряно около $295 млн), торги
приостановлены; данные о перезапуске противоречивы.

**Не проверено:** **GRVT**, **Pacifica** (страницы условий недоступны или не найдены), ряд
мелких площадок из списка DefiLlama (раздел 6), центральные биржи вроде Bybit.

Что это значит для лаборатории: площадок для данных и для будущего исполнения больше, чем
одна Hyperliquid. Выбирать среди доступных — по ликвидности, истории данных и механике
(таблица ниже). Цитаты оговорок ниже оставлены дословно: если условия какой-то площадки
поменяются и Беларусь появится в списке, это видно сразу.

## 2. Сводная таблица

Обозначения доступа: **ЯВНО** — Беларусь названа в запрете (закрыта); **САНКЦ.** — общая санкционная оговорка без списка стран, Беларусь не названа (по правилу владельца — доступна); **НЕЯСНО** — Беларусь не упомянута (доступна); **НЕТ/Н.П.** — не работает / не проверено.
Объём — торги за 24 часа в долларах по данным API самой площадки (сумма мной, 2026-10-25), «ОИ» — объём открытых позиций по DefiLlama (если нет — по API, помечено). История — с какой даты API отдаёт данные по BTC.

| Площадка | Доступ | Оператор · право по условиям | KYC | Продукты | История данных (ставка финансирования) | Объём 24ч / ОИ |
|---|---|---|---|---|---|---|
| Hyperliquid | САНКЦ. | Hyperliquid Corp. · право Англии и Уэльса, арбитраж LCIA (Лондон) | в условиях нет (кошелёк без хранения средств) | бессрочные контракты (234 в основном списке), спот, рынки исходов, рынки сторонних разработчиков (HIP-3), заём | свечи с 2020-08-19 (до запуска — вероятно, подгруженные), ставка с 2023-05-12, **раз в час** | $4,6 млрд / $7,6 млрд (по API $11,1 млрд) |
| Aster | НЕЯСНО | в условиях оператор не назван · право Гонконга | «ограниченная проверка личности может вводиться» | бессрочные контракты (637 тикеров), спот, «Shield», рынки прогнозов, плечо 1001x | свечи с 2021-09-01, ставка с 2021-08-27, **раз в 1/2/4/8 часов по символам** (0,0001 — нижняя граница формулы, не пробел; см. 7.3) | $1,0 млрд / $1,41 млрд |
| Backpack | НЕЯСНО | Trek Labs Ltd FZE (ОАЭ, VARA, только спот) и Trek Labs, Inc. (США) · право — в нечитаемом соглашении | **обязателен** | спот, бессрочные контракты (103), маржа, заём, акции, рынки прогнозов | свечи BTC с 2025-01-05, ставка с 2025-01-14, **раз в час** | $86 млн / ≈$112 млн (по API) |
| dYdX (v4) | НЕЯСНО | dYdX Trading Inc. · право Калифорнии, арбитраж JAMS | в условиях нет | бессрочные контракты (296 рынков), спот | свечи с 2023-11-13, ставка с 2023-10-29, **раз в час** | $33 млн / $50,6 млн |
| GMX | САНКЦ. | «GMX.io» · право Багамских островов | в условиях нет | бессрочные контракты (крипто, металлы, акции, энергия), обмен, пулы GM; развёртывания EVM и Solana | цены с 2021-10-25; история ставки через простой API не найдена | объём не получен / $59,8 млн (EVM) + $213,1 млн (Solana) |
| Lighter | **ЯВНО** | Elliot Technologies, Inc. · право Флориды | в условиях нет | бессрочные контракты (235), спот (11), рынки на 15 минут, акции | свечи с 2025-01-17, ставка с 2025-01-19, **раз в час** | $1,43 млрд / $530,9 млн (+ $213,1 млн отдельная сеть Robinhood) |
| Paradex | НЕЯСНО (противоречиво) | Paradex Corp. · право Англии и Уэльса, арбитраж SIAC (Сингапур) | в условиях нет | бессрочные контракты, **опционы** | запуск рынка 2023-10-04; свечи от 1 минуты; ставка **8-часовая**, обновляется непрерывно | $10,6 млн / $5,4 млн (контракты) + $100,9 млн (опционы) |
| Drift | НЕТ | — (условия не получены) | — | — | API данных не отвечает | — / нет данных у DefiLlama |
| Orderly | НЕЯСНО | Orderly Network Ltd. · право Панамы | «проверки личности — где применимо» | бессрочные контракты (148); инфраструктура, торговля идёт через интерфейсы-брокеры | свечи с 2023-10-26 (первые значения плоские — искусственные), ставка **раз в 8 часов** | $39,8 млн / $43,4 млн |
| Vertex | НЕТ | свёрнут в июле 2025, команда ушла в Ink | — | — | — | — |
| Nado (преемник Vertex) | **ЯВНО** | Spira Arc Inc. | в условиях нет | бессрочные контракты, спот, маржа | свечи с 2021-12-23 (причину не выяснял), ставка **раз в час** | $268,9 млн / $37,7 млн |
| Apex Omni | НЕЯСНО | APEX DAO Inc. · право Панамы | в условиях нет | бессрочные контракты (139), акции, рынки прогнозов, события | ставка с ≈2024-12-27, **раз в час**, свечи от 1 минуты | $369,4 млн / $75,2 млн (по API $145,0 млн) |
| edgeX | НЕЯСНО | edgeX Trading Inc. | в условиях нет | бессрочные контракты (191, в т.ч. золото, серебро, медь, нефть) | **только с ≈2026-05-09** (новые контракты в USDC), ставка **раз в 4 часа** | $473,8 млн / $132,8 млн (по API $262,8 млн) |
| Bluefin Pro | **ЯВНО** | Bluefin Labs Inc. | в условиях нет | 8 бессрочных контрактов | свечи с 2025-07-18, ставка раз в час | $0,6 млн / $0,9 млн |
| Jupiter Perps | САНКЦ. | Block Raccoon S.A. · право Панамы | «может потребовать» | бессрочные контракты только SOL, ETH, BTC против общего пула JLP; спот | **простого API истории нет** (только ончейн-аккаунты) | объём не получен / $45,0 млн |
| Extended | НЕЯСНО | X10 Ltd · право не найдено | в условиях нет (есть «проверка транзакций» KYT) | бессрочные контракты (407, в т.ч. рынки традиционных активов) | свечи с 2024-07-26, ставка с 2025-07-18, **раз в час** | $230,2 млн / $98,9 млн (по API $199,7 млн) |
| Variational | НЕЯСНО (противоречиво) | MCMC Research Corporation, S.A. · право Панамы | в условиях нет | бессрочные контракты и «обмены» (582 рынка), рынки традиционных активов, до IPO | **истории нет** (API даёт только текущую сводку); ставка раз в час | $759,4 млн / $942,5 млн (по API $1,88 млрд двусторонне) |
| GRVT | Н.П. | — | — | бессрочные контракты (184 по API) | свечи с ≈2024-12; ставка раз в 8 часов | $506,7 млн (сумма по 184 тикерам) / $432,4 млн |
| Pacifica | Н.П. | — | — | — | — | — / $62,5 млн |

«ОИ по DefiLlama» берётся из бесплатного API (`https://api.llama.fi/overview/open-interest`); объём торгов по DefiLlama — платный (`Upgrade to the paid API plan`), поэтому объёмы взяты из API самих площадок.

## 3. Разделы по площадкам

В каждой цитате — слова из условий дословно (английский оригинал); перевод цитаты сделал бы её недостоверной.

### 3.1 Hyperliquid — САНКЦИОННАЯ ОГОВОРКА (справочная точка владельца)

- **Оператор интерфейса:** Hyperliquid Corp. Условия обновлены 2026-06-15. Право Англии и Уэльса, арбитраж LCIA в Лондоне (п. 11.4).
- **Что регулирует документ:** сайт-интерфейс `app.hyperliquid.xyz`, а не саму сеть («The Interface is not the exclusive means of accessing Hyperliquid», п. 1.1). Я фиксирую только то, что сказано про интерфейс.
- **Цитаты (п. 1.6):**
  > «(b) persons or entities who reside in, are located in, are incorporated in, or have a registered office in jurisdictions subject to applicable economic and trade sanctions or export control laws and regulations (collectively, “Restricted Territories”)»

  > «(c) citizens of Restricted Territories, regardless of their location.»
- **Почему отношу к санкционной оговорке:** Беларусь под действием (а) санкций ЕС — Регламент Совета (ЕС) № 765/2006 с изменениями (раздел 4.1), (б) экспортного контроля США для России и Беларуси — 15 CFR § 746.8 «Sanctions against Russia and Belarus» (раздел 4.2). Владелец — и резидент, и гражданин, то есть подпадает под оба пункта (b) и (c).
- **Что неясно:** условия не называют страны и не определяют «applicable»; Hyperliquid списка не публикует. Прочтение выше — буквальное, не подтверждённое практикой площадки.
- **Блокировка сайта:** условия допускают «technical, operational or other measures» для рынков исходов; из Нидерландов сайт открылся без блокировки. Как поведёт себя сайт при подключении из Беларуси — не проверялось и не должно.
- **KYC:** в условиях нет; нужен только кошелёк без хранения средств (п. 1.4).
- **Продукты:** бессрочные контракты, спот, рынки исходов (вид рынков прогнозов, п. 1.7), рынки, созданные сторонними разработчиками (HIP-3; у DefiLlama отдельной строкой «tradeXYZ», ОИ $3,87 млрд — работает на сети Hyperliquid), заём под залог HYPE/BTC (баннер сайта).
- **Данные:** публичный POST `https://api.hyperliquid.xyz/info`: `candleSnapshot` (по документации — только **последние 5000 свечей** на запрос), `fundingHistory` (первая запись по BTC — 2023-05-12, шаг тогда 8 часов), `metaAndAssetCtxs` (объём/ОИ). Ставка финансирования выплачивается **раз в час** (одна восьмая от 8-часовой ставки).
  Первая дневная свеча BTC датирована 2020-08-19, то есть **раньше запуска биржи** — это, вероятно, подгруженная история; считать её историей именно Hyperliquid без проверки нельзя.
- **Ликвидность на 2026-10-25:** сумма `dayNtlVlm` по 234 рынкам основного списка — $4,57 млрд; ОИ по DefiLlama $7,57 млрд (по сумме `openInterest × markPx` из API — $11,06 млрд; методики различаются).
- **Источники:** `https://app.hyperliquid.xyz/terms` · `https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding` · `https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint` · `https://api.llama.fi/overview/open-interest`.

### 3.2 Aster — НЕЯСНО

- **Оператор:** в условиях не назван (в тексте — «Aster»); право Гонконга. Условия от 2026-02-25.
- **Цитаты (п. 2.1):**
  > «the United States of America, Canada, the United Kingdom, China, North Korea, Russia, Ukraine, Cuba, Iran, Venezuela, or Syria. This prohibition extends to any jurisdiction or territory that is subject to comprehensive country-wide or regional economic sanctions»

  > «We may implement limited identity verification procedures to comply with applicable regulations.» (п. 2.3)
- **Почему неясно:** Беларусь не названа, а общая оговорка — про «всеобъемлющие (comprehensive) санкции на всю страну». Программа США по Беларуси — по спискам, не всеобъемлющая (раздел 4.2); ЕС — широкие отраслевые ограничения, но является ли это «comprehensive», условия не говорят. Отдельный пункт «Prohibited Person» касается тех, кто лично в санкционных списках; принадлежность владельца к спискам я не проверял и не могу утверждать.
- **Блокировка сайта:** в коде страницы `https://www.asterdex.com/en` есть тексты баннера «You're accessing ASTER from a restricted jurisdiction. Only closing positions and withdrawals are available» — значит, блокировка по IP есть; список стран не опубликован.
- **Продукты (меню документации):** бессрочные контракты, Shield Mode, рынки прогнозов, спот, субсчета, плечо 1001x, сделки вне биржи; по строкам интерфейса — также токенизированные акции («bStocks») и контракты на товары.
- **Данные:** `https://fapi.asterdex.com/fapi/v1/klines` (BTCUSDT с 2021-09-01), `/fapi/v1/fundingRate` (с 2021-08-27; 0,0001 — нижняя граница формулы, константой были только первые десять записей, см. 7.3), `/fapi/v1/fundingInfo` (интервал по символам: 1, 2, 4 или 8 часов; проверено 2026-10-25), `/fapi/v1/ticker/24hr`. Доступ без ключа.
- **Ликвидность:** сумма `quoteVolume` по 637 тикерам — $1,00 млрд; ОИ по DefiLlama $1,41 млрд.
- **Источники:** `https://docs.asterdex.com/resources/terms-and-conditions` · `https://www.asterdex.com/en` · `https://docs.asterdex.com/`.

### 3.3 Backpack — НЕЯСНО (главный документ не прочитан)

- **Оператор:** Trek Labs Ltd FZE — лицензия Дубайского регулятора VARA (спот, «фьючерсы и маржа в ОАЭ недоступны»); Trek Labs, Inc. — раздел «US Licenses» (по состоянию на октябрь 2026). Какое юрлицо обслуживает бессрочные контракты для остальных — на проверенных страницах не сказано.
- **Справочная страница «Supported Regions»:**
  > «We currently do not serve residents of the following countries and regions:»

  далее перечислены Северная Корея, Иран, Куба, Крым, Севастополь, Луганская и Донецкая области; Беларуси в списке нет. Отдельно: «Canada, Japan, New Zealand» — «not currently able to serve».
- **Почему неясно:** справочная страница — не договор. Пользовательское соглашение лежит файлом на Google Drive и не читалось автоматически (страница «Loading»). Старые копии на стороннем сайте не использую — они не первичные.
- **KYC:** обязателен («Why KYC is Required?»), то есть Беларусь будет указана в паспорте — любая неточность становится нарушением условий.
- **Продукты:** спот, бессрочные контракты (103 из 191 рынка), маржа, заём, акции и токенизированные акции, рынки прогнозов (раздел условий на сайте).
- **Данные:** `https://api.backpack.exchange/api/v1/klines` (диапазон запроса ограничен, иначе ошибка «Time range … too long»; BTC_USDC_PERP с 2025-01-05), `/api/v1/fundingRates?symbol=…&limit=1000&offset=…` (с 2025-01-14, раз в час), `/api/v1/tickers`, `/api/v1/openInterest`.
- **Ликвидность:** сумма `quoteVolume` по 140 тикерам, оканчивающимся на `_PERP`, — $86 млн; ОИ ≈ $112 млн (мой расчёт: количество × последняя цена). В рейтинге DefiLlama по ОИ Backpack нет.
- **Источники:** `https://support.backpack.exchange/exchange/exchange-account/identity-verification/supported-regions.md` · `https://support.backpack.exchange/legal/vara-disclosures/vara-license-information.md` · `https://support.backpack.exchange/legal/general-legal/user-agreement.md` · `https://support.backpack.exchange/legal/us-licenses-registrations-and-regulatory-disclosures.md`.

### 3.4 dYdX (v4, dYdX Chain) — НЕЯСНО

- **Оператор:** dYdX Trading Inc. («DTI»); право Калифорнии, арбитраж JAMS (Сан-Франциско). Условия от 2026-03-12.
- **Цитата (определение запрещённой территории):**
  > «“Restricted Territory” means Iran, Cuba, North Korea, the regions of Crimea, Donetsk or Luhansk, or any other country or region that is the subject of comprehensive country-wide or region-wide economic sanctions by the United States»

  Отдельно для бессрочных контрактов условия закрывают «any jurisdiction where the dYdX perpetuals trading software is ineligible for use including without limitation the United States of America, Canada and the United Kingdom» — открытая формула без списка; Беларусь в ней не названа.
- **Почему неясно:** Беларусь не названа, санкции США по ней не всеобъемлющие (раздел 4.2).
- **KYC, блокировка:** KYC в условиях нет; в условиях упомянуты «default detection and prevention mechanisms» для запрещённых лиц.
- **Продукты:** бессрочные контракты и спот (в условиях — «perpetuals and spot trading software»); 296 рынков.
- **Данные:** индексатор `https://indexer.dydx.trade/v4/candles/perpetualMarkets/BTC-USD?resolution=1DAY` (с 2023-11-13), `/v4/historicalFunding/BTC-USD` (почасовая, с 2023-10-29), `/v4/perpetualMarkets` (объём, ОИ).
- **Ликвидность:** `volume24H` по рынкам — $33,5 млн; ОИ $50,5 млн (DefiLlama «dYdX V4» $50,55 млн совпадает).
- **Источники:** `https://dydx.exchange/v4-terms` · `https://indexer.dydx.trade/v4/perpetualMarkets`.

### 3.5 GMX — САНКЦИОННАЯ ОГОВОРКА

- **Оператор:** в условиях назван как «GMX.io» (информационный сайт протокола), право Багамских островов. Условия от 2026-08-20. Развёртывания: GMX EVM (Arbitrum, Avalanche, MegaETH) и GMX Solana — у DefiLlama отдельные строки.
- **Цитата (п. 1.1.4):**
  > «You are not a resident, national, or agent of any country to which the United States, the United Kingdom, or the European Union embargoes goods or imposes similar sanctions»
- **Почему так:** ЕС вводит против Беларуси запреты на поставку товаров (Регламент 765/2006, раздел 4.1) — подпадает под «embargoes goods». Дополнительно запрещены граждане США («You are not a U.S. Person») и обход с помощью VPN.
- **KYC:** в условиях нет.
- **Продукты:** бессрочные контракты (заголовок сайта: «Crypto, Metals, Stocks & Energy»), обмен, пулы ликвидности GM.
- **Данные:** `https://arbitrum-api.gmxinfra.io/prices/candles?tokenSymbol=BTC&period=1d` (1827 дневных свечей, с 2021-10-25). Историю ставки финансирования простым запросом получить не удалось (в API есть только текущее состояние рынков `/markets/info`; история — через индексатор, не проверялось).
- **Ликвидность:** ОИ по DefiLlama — EVM $59,8 млн, Solana $213,1 млн; объём не получен.
- **Источники:** `https://gmx.io/terms-and-conditions` · `https://arbitrum-api.gmxinfra.io/prices/candles?tokenSymbol=BTC&period=1d`.

### 3.6 Lighter — ЯВНО ЗАПРЕЩЕНО

- **Оператор:** Elliot Technologies, Inc.; право штата Флорида. Условия от 2025-12-29.
- **Цитата:**
  > «OUR SERVICES WERE NOT DEVELOPED FOR, AND ARE NOT AVAILABLE TO PERSONS OR ENTITIES WHO RESIDE IN, ARE LOCATED IN … NORTH KOREA, RUSSIA, UKRAINE, CUBA, IRAN, VENEZUELA, SUDAN, BELARUS, MYANMAR OR SYRIA.»
- **Охват:** «Services» включают сайт, протокол и **API** («Interface, Protocol, API, the LIT Token»).
- **KYC:** в условиях нет.
- **Продукты:** бессрочные контракты (235), спот (11), короткие рынки «вверх/вниз» на 15 минут, контракты на акции.
- **Данные:** `https://mainnet.zklighter.elliot.ai/api/v1/candles` (BTC с 2025-01-17), `/api/v1/fundings` (с 2025-01-19, раз в час), `/api/v1/exchangeStats`, `/api/v1/orderBooks`. Ответ на запрос свечей ограничен 500 строк.
- **Ликвидность:** `daily_usd_volume` $1,43 млрд; ОИ по DefiLlama $530,9 млн (отдельно «Lighter Robinhood Perps» $213,1 млн).
- **Источники:** `https://lighter.xyz/terms` · `https://mainnet.zklighter.elliot.ai/api/v1/exchangeStats`.

### 3.7 Paradex — НЕЯСНО (условия противоречат сами себе)

- **Оператор:** Paradex Corp.; право Англии и Уэльса, арбитраж SIAC (Сингапур). Условия от 2026-02-15.
- **Цитата (страница Restricted Persons):**
  > «the subject of economic sanctions or other trade or export restrictions imposed or administered by the United States, the European Union, the United Kingdom, Singapore or the United Nations. These restricted territories are currently:»

  Дальше перечислены Крым, Донецк, Луганск, Севастополь, Куба, Иран, Северная Корея, Сирия, Канада, США и территории США. **Беларуси в перечне нет.**
- **Почему «противоречиво»:** общая фраза буквально описывает любую территорию под экономическими санкциями ЕС (Беларусь под ними — раздел 4.1), а перечень «сейчас» её не содержит. Какая часть главнее, из условий не следует. Условия: «PARADEX DOES NOT MAKE EXCEPTIONS».
- **Продукты:** бессрочные контракты и **опционы**.
- **Данные:** `https://api.prod.paradex.trade/v1/markets/klines?symbol=BTC-USD-PERP&resolution=60` (минимальный шаг 1 минута, максимальный в запросе — 60; рынок открыт 2023-10-04), `/v1/funding/data` (поле `funding_period_hours: 8`), `/v1/markets/summary?market=ALL`.
- **Ликвидность:** сумма `volume_24h` по бессрочным — $10,6 млн; ОИ DefiLlama: контракты $5,4 млн, опционы $100,9 млн.
- **Источники:** `https://paradex.trade/terms-of-service` · `https://www.paradex.trade/restricted-persons`.

### 3.8 Drift — НЕ РАБОТАЕТ ШТАТНО

- **Состояние:** 2026-04-01 взлом на сумму около $295 млн, приостановка основных функций; план перезапуска «в 2 квартале 2026» как биржи только бессрочных контрактов, с возмещением жетонами (статья от 2026-05-05). Утверждение, что платформа переименована в «Velocity» с закрытым бета-тестом (по данным партнёрского сайта, не первичный источник), я **не подтвердил**.
- **Условия и API:** условия отдельной страницей не получены (приложение отдаёт интерфейс торговли, документ не показывается); API данных `data.api.drift.trade` и `mainnet-beta.api.drift.trade` 2026-10-25 не отвечают (ошибка соединения). У DefiLlama ОИ Drift — пусто.
- **Источники:** `https://decrypt.co/366897/how-solana-exchange-drift-repay-users-295-million-crypto-hack` · `https://www.drift.trade/` · `https://app.drift.trade/terms`.

### 3.9 Orderly — НЕЯСНО (для ядра; интерфейсы-брокеры не проверены)

- **Оператор:** Orderly Network Ltd.; право Республики Панама (п. «Governing law»).
- **Цитата (п. 4, «Prohibited Locations»):**
  > «(b) a citizen, resident, or organized in a jurisdiction or territory that is the subject of comprehensive country-wide, territory-wide, or regional economic sanctions by the United States (including, without limitation, Cuba, Democratic People’s Republic of Korea, Iran)»
- **Почему неясно:** Беларусь не названа, «всеобъемлющие» санкции — только США (раздел 4.2). Пункт (a) про лиц в санкционных списках.
- **Важная оговорка:** Orderly — инфраструктура; торгуют через **интерфейсы-брокеры** (в т.ч. WOOFi Pro и другие), у каждого свои условия. Их я не проверял.
- **KYC:** «may conduct … identity verification and authentication checks where applicable».
- **Данные:** `https://api-evm.orderly.org/v1/tv/history?symbol=PERP_BTC_USDC&resolution=1D` (с 2023-10-26; первые значения плоские $34 500 — искусственные), `/v1/public/funding_rate_history` (шаг 8 часов), `/v1/public/futures`.
- **Ликвидность:** `24h_amount` по 148 рынкам — $39,8 млн; ОИ DefiLlama $43,4 млн.
- **Источники:** `https://orderly.network/docs/introduction/terms-of-service` · `https://api-evm.orderly.org/v1/public/futures`.

### 3.10 Vertex и Nado — Vertex свёрнут; Nado: ЯВНО ЗАПРЕЩЕНО

- **Vertex:** в июле 2025 объявил о сворачивании развёртываний на Arbitrum, команда и технологии перешли к Ink Foundation (The Block, 2025-07-08). Источник: `https://www.theblock.co/news/ecosystems/2025-07-08-vertex-to-sunset-vrtx-token-move-perp-dex-onto-kraken-backed-ink-layer-2-361570`.
- **Nado:** оператор Spira Arc Inc.; условия от 2026-08-26.
- **Цитаты:**
  > «you do not reside in, are not located or incorporated in and do not have a registered office in (i) Belarus, Cuba, Iran, North Korea, Russia, or Ukraine»

  Во вступлении: доступ закрыт тем, кто «RESIDE IN, ARE CITIZENS OF, ARE LOCATED IN …» запрещённой территории.
- **Продукты:** бессрочные контракты, спот, маржа с единой маржей. Ставка раз в час.
- **Данные:** `POST https://api.prod.nado.xyz/archive/v1` с `candlesticks` (BTC-PERP, `product_id: 2`, с 2021-12-23 — причина столь ранней даты не выяснялась), `funding_rate` и «Funding Rate History» (по документации), `GET /archive/v2/tickers?market=perp`. Требуется заголовок со сжатием.
- **Ликвидность:** сумма `quote_volume` по 82 тикерам — $268,9 млн; ОИ DefiLlama $37,7 млн.
- **Источники:** `https://www.nado.xyz/legal-docs/terms-of-use` · `https://docs.nado.xyz/developer-resources/api/archive-indexer/candlesticks.md` · `https://docs.nado.xyz/core/funding-rates.md`.

### 3.11 Apex Omni — НЕЯСНО

- **Оператор:** APEX DAO Inc.; право Республики Панама. Условия от 2026-07-27.
- **Цитата:**
  > «Iran, Cuba, North Korea, Syria, Myanmar (Burma), the regions of Crimea, Donetsk or Luhansk, or any other country or region that is the subject of comprehensive country-wide or region-wide economic sanctions by the United States»

  Запрещены также «any jurisdiction where the APEX Software is ineligible for use (including without limitation the United States of America)».
- **Почему неясно:** Беларусь не названа, санкции США по ней не всеобъемлющие (раздел 4.2). Текст совпадает с шаблоном dYdX/edgeX.
- **KYC:** в условиях нет. Продукты: 139 бессрочных, акции, рынки прогнозов, события.
- **Данные:** `https://omni.apex.exchange/api/v3/klines?symbol=BTCUSDT&interval=60`, `/api/v3/history-funding?symbol=BTC-USDT` (с ≈2024-12-27, раз в час), `/api/v3/ticker`, `/api/v3/symbols`.
- **Ликвидность:** сумма `turnover24h` — $369,4 млн; ОИ DefiLlama $75,2 млн (по API $145,0 млн).
- **Источники:** `https://apex-pro.gitbook.io/apex-pro/about/terms-of-use.md` · `https://omni.apex.exchange/api/v3/symbols`.

### 3.12 edgeX — НЕЯСНО

- **Оператор:** edgeX Trading Inc. (право в проверенном тексте не найдено). Блокированы США и Канада.
- **Цитата:**
  > «you must not be a resident, citizen or agent of, or incorporated in, and do not have a registered office in Iran, Cuba, North Korea, Syria, Myanmar (Burma), the regions of Crimea, Donetsk or Luhansk, or any other country or region that is the subject of comprehensive country-wide or region-wide economic sanctions by the United States»
- **Почему неясно:** как у Apex — Беларусь не названа, «всеобъемлющих» санкций США на неё нет.
- **Данные:** `https://edgex-prod-v2.edgex.exchange/api/v2/public/quote/getKline`, `/funding/getFundingRatePage`. **Важно:** старые контракты в USDT (`10000001`) пусты; в ответах только новые контракты в USDC (`30000001`) — свечи ≈ с 2026-05-12, ставка с ≈2026-05-09, интервал **4 часа** (`fundingRateIntervalMin: 240`). Для длинной истории не годится.
- **Продукты:** 191 контракт, в т.ч. золото, серебро, медь, нефть (интерфейс).
- **Ликвидность:** сумма `value` — $473,8 млн; ОИ DefiLlama «edgeX V2» $132,8 млн (по API $262,8 млн).
- **Источники:** `https://edgex-1.gitbook.io/edgeX-documentation/about/terms-of-use.md` · `https://edgex-1.gitbook.io/edgeX-documentation/api-v2/public-api/funding-api.md`.

### 3.13 Bluefin — ЯВНО ЗАПРЕЩЕНО (но рынок мал)

- **Оператор:** Bluefin Labs Inc. Условия от 2024-10-21 (давние: возможно, не отражают нынешний продукт Bluefin Pro).
- **Цитата (п. (v)):**
  > «you are not a resident, national, or agent of, located in, incorporated or otherwise formed in, or have a registered office in, Antigua and Barbuda, Algeria, Bangladesh, Bolivia, Belarus, …»
- **Продукты и данные:** 8 бессрочных контрактов; `https://api.sui-prod.bluefin.io/v1/exchange/candlesticks` (с `type=last`; с 2025-07-18), `/fundingRateHistory`, `/tickers`.
- **Ликвидность:** объём за сутки $0,57 млн, ОИ $0,89 млн — для исследования бесполезно.
- **Источники:** `https://bluefin.io/terms-of-use` · `https://api.sui-prod.bluefin.io/v1/exchange/tickers`.

### 3.14 Jupiter Perps — САНКЦИОННАЯ ОГОВОРКА

- **Оператор:** Block Raccoon S.A.; право Панамы.
- **Цитата (разд. «Prohibited Localities»):**
  > «Jupiter does not interact with digital wallets located in, established in, or a resident of [список стран] or any other state, country or region that is subject to sanctions enforced by the United States, the United Kingdom or the European Union.»

  В списке Беларуси нет, но «любая страна, к которой применяются санкции США, Великобритании или ЕС» подходит (разделы 4.1–4.3).
- **KYC:** «We reserve the right to conduct “Know Your Customer” … checks».
- **Продукты:** бессрочные контракты только SOL, ETH, BTC против общего пула JLP (документация: счета Custody — SOL, ETH, BTC, USDC, USDT, JupUSD). История — только ончейн-счета; **простого API для истории нет**.
- **Ликвидность:** ОИ DefiLlama $45,0 млн.
- **Источники:** `https://developers.jup.ag/docs/legal/terms-of-use.md` · `https://developers.jup.ag/docs/perps/index.md`.

### 3.15 Extended — НЕЯСНО

- **Оператор:** X10 Ltd (юрисдикция в тексте не названа). Условия обновлены «08/10/2026» (формат даты в тексте не раскрыт).
- **Цитата (страница Restricted Countries):**
  > «“Restricted Persons” includes any individual or entity residing in, located in, incorporated in, or with a registered office in any jurisdiction in which access to or use of the Extended Protocol is restricted, including without limitation:»

  Список: Афганистан, Канада, Китай (Гонконг), Куба, Иран, Ирак, Йемен, Северная Корея, Россия, Сирия, Сейшелы, Украина (Крым, Донецк, Херсон, Луганск, Запорожье), Великобритания, США. **Беларуси нет**; последняя строка — «any other individual or entity whose access … would be contrary to any applicable law or regulatory requirement».
- **Почему неясно:** список «including without limitation» открыт, Беларусь не названа; общая оговорка «contrary to applicable law» зависит от толкования.
- **Продукты:** 407 рынков, включая традиционные активы. Ставка раз в час.
- **Данные:** `https://api.starknet.extended.exchange/api/v1/info/candles/BTC-USD/trades?interval=P1D` (с 2024-07-26), `/info/BTC-USD/funding` (с 2025-07-18; 1000 записей на запрос), `/info/markets`.
- **Ликвидность:** сумма `dailyVolume` — $230,2 млн; ОИ DefiLlama $98,9 млн (по API $199,7 млн).
- **Источники:** `https://docs.extended.exchange/extended-resources/legal/restricted-countries.md` · `https://docs.extended.exchange/extended-resources/legal/terms-of-use.md`.

### 3.16 Variational — НЕЯСНО (противоречиво)

- **Оператор:** MCMC Research Corporation, S.A.; право Панамы. Условия от 2026-09-11; страница Restricted Persons от 2026-04-20. Сам сайт `omni.variational.io` показал страницу проверки Cloudflare («Just a moment…») — её я не обходил, условия прочитаны из документации.
- **Цитата:** та же формула, что у Paradex:
  > «the subject of economic sanctions or other trade or export restrictions imposed or administered by the United States, the European Union, the United Kingdom, Singapore or the United Nations. These restricted territories are currently:»

  Перечень: Крым, Донецк, Луганск, Севастополь, Куба, Иран, Северная Корея, **Судан**, Сирия, Канада, США, территории США, **Тайвань**. Беларуси нет. В п. 2.G прямо сказано о возможной блокировке по месту нахождения.
- **Продукты:** бессрочные контракты и «обмены» (582 рынка), рынки традиционных активов, контракты до IPO.
- **Данные:** `https://omni-client-api.prod.ap-northeast-1.variational.io/metadata/v2/stats` — только текущая сводка, **без истории**.
- **Ликвидность:** `total_volume_24h` $759,4 млн; `open_interest` $1,88 млрд (двусторонний; DefiLlama $942,5 млн).
- **Источники:** `https://docs.variational.io/legal/terms-of-use.md` · `https://docs.variational.io/legal/restricted-persons.md` · `https://docs.variational.io/technical-documentation/api.md`.

### 3.17 GRVT и Pacifica — не проверены

- **GRVT:** страницы условий не нашлись (`grvt.io/terms-of-use` — 404, ссылок «Terms» на главной нет, сервер отвечает Cloudflare 403 на запросы без браузера). Публичный API данных работает: `POST https://market-data.grvt.io/full/v1/kline` (дневные свечи BTC с ≈2024-12), `/full/v1/funding` (интервал 8 часов), `/full/v1/ticker`; 184 бессрочных контракта; объём (сумма по тикерам) $506,7 млн, ОИ DefiLlama $432,4 млн.
- **Pacifica:** условия не найдены (`pacifica.fi/terms` — 404; документация без юридических страниц); ОИ DefiLlama $62,5 млн.

## 4. Санкционный фон

### 4.1 Евросоюз: Регламент Совета (ЕС) № 765/2006

Статья 1u, пункт 2 (редакция Регламента (ЕС) 2025/2041 от 2025-10-23, вступил в силу на следующий день после публикации):

> «It shall be prohibited to provide, directly or indirectly, the following services to Belarusian nationals or natural persons residing in Belarus, or to legal persons, entities or bodies established in Belarus: (a) crypto-asset services, as defined in Regulation (EU) 2023/1114»

Пункт 3 (в редакции Регламента (ЕС) 2026/1846 от 2026-07-23) запрещает ещё и позволять гражданам и резидентам Беларуси владеть, контролировать или входить в органы управления компании, учреждённой в государстве-члене и оказывающей услуги с криптоактивами: для услуг хранения и счетов — с 2025-03-26, для всех остальных услуг по определению MiCA — **с 2026-08-25**. Пункт 4: запреты не применяются к гражданам государств-членов ЕС, ЕЭЗ и Швейцарии и к лицам с видом на жительство там:

> «Paragraphs 1, 2 and 3 shall not apply to nationals of a Member State, of a country member of the European Economic Area or of Switzerland, or to natural persons having a temporary or permanent residence permit …»

Владелец под исключение не подпадает (гражданин и резидент Беларуси).

История: запрет услуг хранения, счетов и кошельков введён Регламентом (ЕС) 2025/392 от 2025-02-24, расширен на **все** услуги с криптоактивами, платёжные услуги и электронные деньги Регламентом (ЕС) 2025/2041. Регламент 2026/513 от 2026-04-23 добавил запрет операций с юрлицами, оказывающими услуги с криптоактивами и учреждёнными в Беларуси, и с указанными в приложении Беларусью криптоактивами и цифровыми валютами центрального банка.

**Кого связывает.** Статья 10: Регламент применяется «within the territory of the Community», к гражданам государств-членов, к юрлицам, учреждённым по праву государства-члена, и

> «to any legal person, entity or body in respect of any business done in whole or in part within the Community»

Значит, запрет прямо обязывает тех, кто подпадает под право ЕС: юрлица, учреждённые в государстве-члене, граждан государств-членов и тех, кто ведёт бизнес целиком или частично на территории ЕС. Среди проверенных площадок ни у одной в условиях не назван оператор из ЕС (место регистрации у большинства в тексте не названо; Багамы, Гонконг, Панама, Флорида, Калифорния в таблице — это **применимое право договора**, а не обязательно место регистрации). Применимость запрета к децентрализованным протоколам и интерфейсам — вопрос права; я его не решаю.

Источники (тексты через Publications Office ЕС, доступ 2026-10-25):
- `https://publications.europa.eu/resource/celex/32025R0392` (Регламент 2025/392);
- `https://publications.europa.eu/resource/celex/32025R2041` (Регламент 2025/2041);
- `https://publications.europa.eu/resource/celex/32026R1846` (Регламент 2026/1846);
- `https://publications.europa.eu/resource/celex/32026R0513` (Регламент 2026/513);
- сводная редакция на 2026-07-24: `https://publications.europa.eu/resource/celex/02006R0765-20260724`.

(Запросы шли с заголовком `Accept: application/xhtml+xml`, потому что обычная страница EUR-Lex отдаёт заглушку проверки на бота.)

### 4.2 США: программа OFAC по Беларуси

- Программа **по спискам, а не всеобъемлющая**. Основа: указы 13405 (2006) и 14038 (2021), регламент 31 CFR Part 548. Статья 548.201 блокирует имущество **перечисленных лиц**:
  > «All property and interests in property … of the following persons are blocked and may not be transferred, paid, exported, withdrawn, or otherwise dealt in»

  Запрета обслуживать любых жителей Беларуси она не содержит. Есть ещё ограничение на новый долг Минфина и Банка развития Беларуси (§ 548.202).
- Страница программы OFAC (названия записей): общая лицензия № 12 по воздушным судам (2025-11-04), общая лицензия № 14 (2026-03-26; в названии записи — Belarussian Bank of Development and Reconstruction, по смыслу — Белинвестбанк и ряд других лиц), исключения из списка (2026-09-30 и 2026-10-21) — то есть в последнее время идёт частичное **ослабление**. Содержание исключений и лицензий страница не раскрывает.
- **Экспортный контроль США** (другое от санкций OFAC): 15 CFR § 746.8 «Sanctions against Russia and Belarus» — лицензия на вывоз/передачу в Беларусь или внутри неё любых товаров, ПО и технологий из перечня. К финансовым услугам отношения не имеет, но как раз эти слова («export control laws») использует оговорка Hyperliquid.
- Из этого следует: для американских операторов (Elliot Technologies, dYdX Trading, edgeX Trading и др. «Inc.») сам по себе факт резидентства в Беларуси OFAC-запретом не является, если лицо не в списках. Запреты в их условиях — их собственное решение.

Источники: `https://ofac.treasury.gov/sanctions-programs-and-country-information/belarus-sanctions` · `https://www.ecfr.gov/current/title-31/subtitle-B/chapter-V/part-548` (текст через API eCFR на 2026-10-20, ред. 27.07.2026) · `https://www.ecfr.gov/current/title-15/subtitle-B/chapter-VII/subchapter-C/part-746/section-746.8`.

### 4.3 Великобритания

- Регламент о санкциях по Беларуси 2019 (SI 2019/600, `https://www.legislation.gov.uk/uksi/2019/600`): режим по спискам лиц (замораживание активов) плюс торговые меры. Посредники криптоактивов (cryptoasset exchange provider, custodian wallet provider) упомянуты только в перечне «relevant firms» (рег. 39), которые обязаны сообщать о нарушениях (рег. 38). Запрета обслуживать резидентов Беларуси **по одному только резидентству** я в тексте не нашёл (версия, отданная сайтом 2026-10-25).

### 4.4 ООН

Режима санкций Совета Безопасности ООН против Беларуси я не проверял (страница ООН для автоматического чтения недоступна) — см. раздел 5.

## 5. Что не удалось проверить

1. **Юридическую силу выводов.** Таблица — классификация текстов условий; решения о допустимости принимает юрист. Особенно: буквальное прочтение условий Hyperliquid, GMX, Jupiter как закрывающих Беларусь.
2. **Практику площадок.** Блокируют ли сайты IP из Беларуси и на какой стадии — не проверялось (все запросы шли из Нидерландов; сайты открылись без блокировки). Единственный признак блокировки — тексты баннера Aster.
3. **Backpack: пользовательское соглашение** (файл на Google Drive) не прочитано; GRVT и Pacifica — условий не нашёл; Drift — условия не получены.
4. **Интерфейсы-брокеры Orderly** (WOOFi Pro и др.) — свои условия, не читались.
5. **Охват условий на API.** У Lighter, Aster, Nado, Extended, Bluefin, Variational под «Services» подпадает и API. Я читал только публичные справочные запросы (по заданию); регулярная выгрузка данных для бэктеста — вопрос к юристу.
6. **Режим ООН и применимость санкций ЕС к децентрализованным протоколам.** Не проверялись.
7. **Законы самой Беларуси** (о криптоактивах, валютном контроле, налогах) — вне этого документа.
8. **Объём торгов у GMX и Jupiter**, историю ставки у GMX — простым открытым запросом получить не удалось. DefiLlama отдаёт объёмы по бессрочным контрактам только платно.
9. **Hyperliquid, дневные свечи до запуска** (с 2020-08-19) — происхождение не выяснено.
10. **Из рейтинга DefiLlama по ОИ не рассмотрены:** tradeXYZ (работает на Hyperliquid, $3,87 млрд), Antarctic ($324 млн), QFEX ($160 млн), SoDEX ($151 млн), Vest ($88 млн), Pacifica, RISEx, TxFlow, Ondo Perps, Polymarket Perps, Arcus, Aevo, Ostium, Hibachi, Reya, KiloEx, MYX (`https://api.llama.fi/overview/open-interest`, 2026-10-25). Kalshi и Polymarket — не биржи бессрочных контрактов.

## 6. Как воспроизвести

Условия — страницы по адресам из разделов 3 (на GitBook-документациях доступна версия `.md` по тому же адресу). ОИ — `curl 'https://api.llama.fi/overview/open-interest?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true'`. Данные бирж — публичные запросы из раздела 3 без ключей и без регистрации.

## 7. Площадка для данных лаборатории (T51, 2026-10-25)

Лаборатория получила вторую площадку данных после Hyperliquid и Binance — **Aster**
(`data.source: aster`). Это только данные: ордеров лаборатория не ставит, исполнение на Aster
не строилось. Условия Aster — раздел 3.2: Беларусь в них не названа, по правилу владельца
(пункт 2 `docs/SEARCH_CONSTRAINTS.md`) площадка доступна. Оговорка: условия Aster
перечисляют Россию, но не Беларусь; если список изменится, это видно сразу. Охват условий
на выгрузку данных через API — вопрос к юристу (раздел 5, пункт 5).

### 7.1 Как выбирали

Из короткого списка (dYdX v4, Aster, Paradex) подходит по критериям «не меньше ~18 месяцев
свечей и ставок, без ключа, заметный объём, много рынков, документированный API». Цифры
сняты 2026-10-25 публичными запросами:

| | Aster | dYdX v4 | Paradex |
|---|---|---|---|
| История свечей / ставок | с 2021-09 / с 2021-08 (≈61 месяц) | с 2023-11 / с 2023-10 (≈35 месяцев; ставка равна нулю до 2023-11-24) | рынок с 2023-10 |
| Объём за сутки | **$1,0 млрд**, 111 рынков торгуют больше $100 тыс., 25 — больше $1 млн | $33 млн, **больше $1 млн только у трёх рынков** (BTC, ETH, SOL); у остальных 70 живых — десятки тысяч долларов, у TRX 46 часов из 48 без сделок | $10,6 млн |
| Рынков | 643 (605 в USDT) | 296, из них 73 живых | не считал |
| Закрытые рынки | видны только «сворачиваемые» (36); ранее закрытые API не перечисляет, но отдаёт по имени | **видны все: 223 рынка в статусе финального расчёта** | не выяснял |
| Фандинг | 1 / 2 / 4 / 8 часов по символам | раз в час | 8 часов |
| API | документирован (GitHub `asterdex/api-docs`), формат как у Binance | документирован, свечи по 1000 | документирован |

**Почему Aster, а не dYdX.** dYdX честнее по списку закрытых рынков, но как площадка он
почти мёртв: ликвидны три рынка, на остальных данные — это редкие сделки и плоские свечи.
Стратегию, которую стенд не может проверить на «живом» объёме, там не проверить, а
нынешний объём не даст и проверки вперёд. У Aster в тридцать раз больше объёма, в
двадцать раз больше рынков, пять лет истории и фандинг с другими правилами (разные
интервалы по символам, потолки от ±0,3% до ±3%, нулевая процентная ставка у акций). **Цена
выбора — выживаемость** (7.3). dYdX остаётся запасным вариантом, если нужен список закрытых
рынков; Paradex отброшен из-за объёма и короткой истории.

### 7.2 Что построено

- `src/qlab/data/sources/aster.py` — свечи (закрытие, объём в единицах монеты, максимум за
  свечу, число сделок) с интервалами 1d / 4h / 1h; история ставок в собственном интервале
  контракта; постраничная выгрузка, пауза между запросами, кэш `data/raw/aster/` с
  возобновлением.
- Источник зарегистрирован в `qlab.data.snapshot`; `data.source: aster` работает в спеке,
  `qlab data fetch --source aster --instruments BTC,ETH,SOL …` строит снимок.
- `venues/aster.yaml` — факты площадки (комиссии с адресом, интервал фандинга, минимальная
  нога $5; исполнение и атомарность двух ног — **неизвестно**, поля пустые).
- Тесты `src/qlab/data/sources/test_aster.py` — на записанных ответах, без сети.

Комиссии по документации (`docs.asterdex.com/trading/perpetuals/fees-and-specs/fees.md`):
мейкер 0; тейкер 0,04% на основных контрактах (зависит от уровня VIP), 0,1% на «группе B»,
1,25 б.п. на контрактах реальных активов (акции, сырьё) с 2026-09-07.

### 7.3 Что не так с этими данными

1. **Выживаемость.** Aster не перечисляет закрытые рынки, поэтому `describe_universe` у него
   пустой, а снимок без списка монет невозможен: нужен явный список, и снимок помечается
   «список неполный» (`universe_complete=False`). Правило честного списка такой снимок не
   пропустит для вывода о кросс-секционной стратегии — так и задумано. Смягчение:
   `aster.best_effort_universe()` перебирает все имена, которые когда-либо были на Binance,
   и оставляет те, что Aster ещё отдаёт: 2026-10-25 это 605 живых и сворачиваемых контрактов
   плюс **57 закрытых** (LUNA, FTT, TOMO и др.; 360 запросов, около 10 минут, дальше кэш).
   Контракт, который был только на Aster, так не найти, поэтому это список для человека,
   а не доказательство полноты.
2. **Закрытый контракт продолжает выдавать пустые свечи** (плоская цена, ноль сделок — так
   ведут себя TON и PAPER в статусе «сворачивается»), а ставка у FTT идёт ещё год после
   последней сделки. Источник обрезает историю закрытого контракта по последней свече со
   сделками и ставку — по тому же моменту; живой контракт с тихими часами не трогается.
3. **Ставка 0,0001 — не пробел.** Формула даёт ровно 0,01% за 8 часов всякий раз, когда
   премия мала (у BTC так выходит треть дней). Постоянной она была только в первых десяти
   записях (2021-08-27 … 08-30). Прежняя заметка («в ранние годы константа») была неверна.
4. **Интервал фандинга меняется от символа к символу и может меняться во времени.** Сетка
   панели — 8 часов; платежи с меньшим интервалом складываются в свой 8-часовой отрезок
   (как у Binance). Для дневных свечей это точно. Для часовых свечей платёж 4-часового
   символа приходит на конец 8-часового отрезка, то есть **до 4 часов позже** срока, и
   пропуск одного платежа на часовых свечах не виден (ноль между отметками).
5. **Только контракты в USDT и без спота.** Контракты в USD1, USDC и «U» (32 на 2026-10-25)
   не берутся; спот Aster не подключён, поэтому стратегия «монета против контракта» (FRAB,
   Bv2) на этой площадке не строится.
6. **Объём не проверен на накрутку.** На стенде для этого есть фильтр `min_daily_volume_usd`,
   и пользоваться им стоит.
7. **Ежедневная запись списка рынков** (`qlab.venue_watch`) для Aster не заведена: новые и
   закрытые контракты пока не отслеживаются, как у Hyperliquid и Binance.
