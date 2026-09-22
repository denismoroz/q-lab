// Russian interface labels for the registry's stored enum values.
//
// The keys are the values as docs/REGISTRY.md defines them; the values are
// what the owner reads. Nothing here changes meaning — an unmapped value
// falls through unchanged rather than being hidden.

export const IDEA_STATUS = {
  candidate: 'кандидат',
  speccing: 'формализация',
  implemented: 'реализована',
  validated: 'провалидирована',
  bench: 'скамейка',
  paper: 'бумажная торговля',
  live: 'в бою',
  rejected: 'отвергнута',
  decayed: 'выдохлась',
  retired: 'выведена',
}

// Terminal states get a muted, distinct treatment: the graveyard is not a
// failure list to be hidden, but it must not read like a live strategy.
export const TERMINAL_STATUSES = new Set(['rejected', 'decayed', 'retired'])

export const STATUS_STYLE = {
  candidate: 'bg-slate-100 text-slate-700 ring-slate-300',
  speccing: 'bg-slate-100 text-slate-700 ring-slate-300',
  implemented: 'bg-sky-50 text-sky-800 ring-sky-300',
  validated: 'bg-sky-50 text-sky-800 ring-sky-300',
  bench: 'bg-indigo-50 text-indigo-800 ring-indigo-300',
  paper: 'bg-emerald-50 text-emerald-800 ring-emerald-300',
  live: 'bg-emerald-100 text-emerald-900 ring-emerald-400',
  rejected: 'bg-stone-100 text-stone-600 ring-stone-300',
  decayed: 'bg-stone-100 text-stone-600 ring-stone-300',
  retired: 'bg-stone-100 text-stone-600 ring-stone-300',
}

export const SHUTDOWN_CAUSE = {
  'edge-decayed': 'эдж выдохся',
  'false-discovery': 'ложное открытие',
  execution: 'исполнение',
  'owner-choice': 'решение владельца',
}

export const PROFILE = {
  carry: 'carry',
  momentum: 'моментум',
  'mean-reversion': 'возврат к среднему',
  arb: 'арбитраж',
  other: 'прочее',
}

export const SOURCE_TYPE = {
  paper: 'статья',
  github: 'github',
  forum: 'форум',
  'venue-event': 'событие площадки',
  graveyard: 'кладбище',
  internal: 'внутренняя',
}

export const ASSET_CLASS = {
  'crypto-perp': 'крипто-перп',
  'crypto-spot': 'крипто-спот',
  defi: 'defi',
  fx: 'fx',
}

export const VERDICT_STAGE = {
  preflight: 'preflight',
  edge: 'эдж',
  tail: 'хвосты',
  profile: 'профиль',
  capacity: 'ёмкость',
  correlation: 'корреляция',
}

export const VERDICT_OUTCOME = {
  passed: 'прошло',
  failed: 'провалено',
  unknown: 'неизвестно',
  measurement: 'измерение',
}

export const TRIAL_STATUS = { ok: 'ok', error: 'ошибка' }
export const ROW_SOURCE = { qlab: 'q-lab', imported: 'импорт' }

export const CARD_PART = {
  driver: 'Драйвер — кто и за что платит',
  signal: 'Сигнал — вход и выход',
  sizing: 'Размер',
  protection: 'Защита хвостов',
  execution: 'Исполнение',
  death: 'Как умрёт',
}

export const CARD_FIELD = {
  who_pays: 'кто платит',
  why: 'почему',
  kill_condition: 'условие смерти драйвера',
  observable: 'наблюдаемая величина',
  entry: 'вход',
  exit: 'выход',
  data_needed: 'нужные данные',
  timeframe: 'горизонт',
  rule: 'правило размера',
  binding_constraint: 'что ограничивает',
  crash: 'от краха',
  pump: 'от пампа',
  costs: 'от костов',
  structural: 'структурная',
  note: 'примечание',
  legs: 'ног',
  atomic_required: 'нужна атомарность',
  venue: 'площадка',
  min_leg_notional_usd: 'мин. нога, $',
  how: 'как',
  visible_in_advance: 'видно заранее',
  data_free: 'данные бесплатны',
  data_source: 'источник данных',
  venue_supported: 'площадка поддержана',
  min_capital_order_of_magnitude_usd: 'порядок капитала, $',
  blockers: 'блокеры',
  graveyard_matches: 'совпадения с кладбищем',
}

export const label = (dictionary, value) =>
  value == null ? null : (dictionary[value] ?? value)
