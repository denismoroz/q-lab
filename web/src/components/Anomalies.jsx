// docs/ANOMALIES.md: the anomaly detector's flags for the idea's latest scanned run and, when the
// investigator agent looked into it, its dossier. The flags decide nothing and the agent changes
// no verdict: both are shown for the owner. Every sentence comes from the API (`anomalies`),
// worded by code from what the scan stored; nothing is computed here.

const STATE_STYLE = {
  fires: 'bg-amber-100 text-amber-900 ring-amber-300',
  quiet: 'bg-slate-100 text-slate-600 ring-slate-300',
  'not defined': 'bg-white text-slate-500 ring-slate-300',
}

const CHECK_STATUS = {
  none: 'проверку агент не предлагал',
  proposed: 'ждёт вашего решения',
  adopted: 'принята: сделана признаком',
  declined: 'отклонена',
}

function FlagRow({ flag }) {
  return (
    <li className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
      <span
        className={`inline-block whitespace-nowrap rounded px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${
          STATE_STYLE[flag.state] ?? STATE_STYLE.quiet
        }`}
      >
        {flag.state_ru}
      </span>
      <span className="text-sm text-slate-700">{flag.sentence}</span>
    </li>
  )
}

function Dossier({ investigation }) {
  const inv = investigation
  return (
    <div className="mt-4 rounded border border-slate-200 bg-white p-3" data-testid="investigation">
      <div className="text-xs uppercase tracking-wide text-slate-500">
        Разбор агентом-следователем (вердикт он не меняет)
      </div>
      <div className="mt-1 text-base font-semibold text-slate-900">{inv.cause_ru}</div>
      <p className="mt-1 text-sm text-slate-700">{inv.summary}</p>
      {inv.evidence.length > 0 && (
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-700">
          {inv.evidence.map((e, i) => (
            <li key={i}>
              {e.text}{' '}
              <span className="font-mono text-xs text-slate-400">
                [вызов {e.call}: «{e.quote}»]
              </span>
            </li>
          ))}
        </ul>
      )}
      {inv.proposals.length > 0 && (
        <>
          <div className="mt-3 text-xs uppercase tracking-wide text-slate-500">что предлагается</div>
          <ul className="mt-1 list-disc space-y-1 pl-5 text-sm text-slate-700">
            {inv.proposals.map((p, i) => (
              <li key={i}>
                {p.text}
                {p.programmer_request ? (
                  <span className="text-xs text-slate-500">
                    {' '}
                    (заявка программисту {p.programmer_request}; слияние только вашей командой
                    `qlab code merge`)
                  </span>
                ) : null}
              </li>
            ))}
          </ul>
        </>
      )}
      <p className="mt-3 text-sm text-slate-700">
        {inv.caught_by.length > 0
          ? `Причину уловили признаки: ${inv.caught_by.join(', ')}.`
          : inv.cause === 'unclear' || inv.cause === 'genuine'
            ? 'Детектору учиться нечему: ошибка не подтверждена.'
            : 'Ни один признак причину не уловил.'}
        {inv.new_check
          ? ` Предложена проверка: ${inv.new_check} (${CHECK_STATUS[inv.new_check_status] ?? ''}).`
          : ''}
      </p>
      <details className="mt-2">
        <summary className="cursor-pointer text-xs text-sky-700">досье целиком</summary>
        <pre className="mt-2 whitespace-pre-wrap text-xs text-slate-700">{inv.dossier}</pre>
      </details>
      <p className="mt-2 text-xs text-slate-500">
        разбор #{inv.id}, {inv.at.slice(0, 10)}, модель {inv.model}, токенов{' '}
        {inv.tokens.toLocaleString('ru-RU')}. Причина проверена кодом по журналу вызовов агента.
      </p>
    </div>
  )
}

export default function Anomalies({ block }) {
  if (!block?.scan) return null
  const { scan, investigation } = block
  const fired = scan.flags.filter((f) => f.state === 'fires').length
  return (
    <section
      className={`rounded border p-4 ${fired ? 'border-amber-400 bg-amber-50' : 'border-slate-200'}`}
      data-testid="anomalies"
    >
      <div className="text-xs uppercase tracking-wide text-slate-500">
        Аномалии — признаки только записываются, на вердикт не влияют
      </div>
      <div className="mt-1 text-lg font-semibold text-slate-900">
        {fired
          ? `Сработало признаков: ${fired} из ${scan.flags.length}`
          : 'Ни один признак не сработал'}
      </div>
      <ul className="mt-2 space-y-1">
        {scan.flags.map((f) => (
          <FlagRow key={f.name} flag={f} />
        ))}
      </ul>
      {investigation && <Dossier investigation={investigation} />}
      <p className="mt-3 text-xs text-slate-500">
        прогон #{scan.trial_id} ({scan.route}), проверено {scan.scanned_at.slice(0, 10)}, границы
        версии {scan.config_version}.
      </p>
    </section>
  )
}
