import { useRef, useState, type FormEvent } from 'react'
import { Link } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, type TStrategyConfig, type TStrategyResult, type TStrategyStatus } from '@/lib/api'
import { QK } from '@/lib/queryKeys'
import { PageHeader } from '@/components/PageHeader'

const buttonClass = 'min-h-10 rounded-btn border border-border px-4 py-2 text-sm font-medium hover:bg-elevated focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-50'
const inputClass = 'w-full min-h-10 rounded-btn border border-border bg-surface px-3 py-2 text-sm text-foreground focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-50'
const panelClass = 'rounded-card border border-border bg-surface p-4 sm:p-5 space-y-4'
const money = (value: number | undefined) => value == null || !Number.isFinite(value) ? '—' : value.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
const errorText = (error: unknown) => error instanceof Error ? error.message : '操作失败，请重试'
const statusLabels: Record<string, string> = {
  pending: '待成交', filled: '已成交', rejected: '已拒绝', stale_data: '数据已过期',
  no_data: '缺少本地数据', missing_data: '缺少本地数据', no_signal: '未触发信号',
  idle: '未触发信号', outside_session: '非交易时段', market_closed: '非交易时段',
  already_processed: '本分钟已评估', duplicate: '本分钟已评估', waiting: '等待可用分钟线',
  insufficient_data: '数据不完整', error: '评估失败', data_timeout: '数据读取超时',
  already_evaluated: '本分钟已评估', order_created: '已创建模拟挂单', evaluated: '评估完成',
}

function seedConfig(status: TStrategyStatus): TStrategyConfig {
  const end = status.coverage.end?.slice(0, 10) || status.defaults.end
  const start = status.coverage.start?.slice(0, 10) || status.defaults.start
  const earliest = new Date(`${end}T00:00:00Z`)
  earliest.setUTCDate(earliest.getUTCDate() - 89)
  const boundedStart = Number.isFinite(earliest.getTime()) ? (start < earliest.toISOString().slice(0, 10) ? earliest.toISOString().slice(0, 10) : start) : start
  return { ...status.defaults, symbol: '000001.SZ', start: boundedStart, end }
}
function validate(config: TStrategyConfig, needsDates: boolean): string {
  if (!/^\d{6}\.(SH|SZ|BJ)$/.test(config.symbol)) return '请输入六位股票代码及交易所后缀，如 000001.SZ'
  if (![config.base_qty, config.trade_qty].every(n => Number.isInteger(n) && n > 0 && n % 100 === 0)) return '底仓和每次做T股数必须是正数且为 100 股的整数倍'
  if (config.trade_qty > config.base_qty) return '每次做T股数不能超过底仓'
  if (!Number.isFinite(config.initial_cash) || config.initial_cash <= 0) return '初始资金必须是有限正数'
  if (![config.rise_pct, config.pullback_pct].every(n => Number.isFinite(n) && n > 0 && n <= 30)) return '上涨和回落阈值须大于 0 且不超过 30%'
  if (!Number.isFinite(config.commission_pct) || config.commission_pct < 0 || config.commission_pct > 0.01) return '佣金比例须在 0 到 0.01 之间'
  if (!Number.isFinite(config.slippage_bps) || config.slippage_bps < 0 || config.slippage_bps > 100) return '滑点须在 0 到 100 个基点之间'
  const validTime = (t: string) => /^\d{2}:\d{2}$/.test(t) && ((t >= '09:31' && t <= '11:30') || (t >= '13:01' && t < '15:00'))
  if (!validTime(config.start_time) || !validTime(config.restore_time) || config.start_time >= config.restore_time) return '开始和回补时间须在交易时段内，且开始时间早于回补时间'
  if (needsDates) {
    const days = (Date.parse(config.end) - Date.parse(config.start)) / 86400000 + 1
    if (!config.start || !config.end || !Number.isFinite(days) || days < 1 || days > 90) return '回测日期须为有效区间，且最多 90 个自然日'
  }
  return ''
}

function NavChart({ nav }: { nav: TStrategyResult['nav'] }) {
  if (!nav.length) return <p className="text-sm text-muted">暂无净值记录。</p>
  const low = Math.min(...nav.map(n => n.nav))
  const high = Math.max(...nav.map(n => n.nav))
  const span = high - low || Math.max(Math.abs(high) * 0.01, 1)
  const y = (v: number) => 140 - (v - low) / span * 100
  const points = nav.map((n, i) => `${70 + (nav.length === 1 ? 320 : i / (nav.length - 1) * 640)},${y(n.nav)}`).join(' ')
  return <figure>
    <svg role="img" aria-label="账户净值曲线" viewBox="0 0 760 190" className="w-full min-h-36 text-secondary">
      <title>账户净值曲线（元，包含应收股息）</title>
      <path d="M70 30V150H710" fill="none" stroke="currentColor" opacity="0.3" />
      <text x="0" y="44" fontSize="12" fill="currentColor">{money(high)}</text>
      <text x="0" y="145" fontSize="12" fill="currentColor">{money(low)}</text>
      <polyline points={points} stroke="currentColor" className="text-accent" strokeWidth="2.5" fill="none" />
      {nav.length === 1 && <circle cx="390" cy={y(nav[0].nav)} r="4" fill="currentColor" className="text-accent" />}
      <text x="70" y="177" fontSize="12" fill="currentColor">{nav[0].date}</text>
      <text x="710" y="177" fontSize="12" fill="currentColor" textAnchor="end">{nav[nav.length - 1].date}</text>
    </svg>
    <figcaption className="text-xs text-muted">账户净值（元），含应收股息；首尾日期见横轴。</figcaption>
  </figure>
}
function ResultPanel({ result }: { result: TStrategyResult }) {
  const { stats } = result
  return <section className={panelClass} aria-label="回测结果">
    <h2 className="font-semibold">回测结果 · {result.params.symbol}</h2>
    <p className="text-xs text-muted">{result.coverage.start} 至 {result.coverage.end} · {result.coverage.sessions} 个交易日 · {result.coverage.bars} 根分钟线 · 运行编号 {result.id}</p>
    <dl className="grid grid-cols-2 gap-4 lg:grid-cols-4">
      {[
        ['期末净资产', money(stats.final_nav)], ['总收益率', `${stats.return_pct.toFixed(2)}%`],
        ['相对买入持有超额盈亏', money(stats.excess_pnl)], ['完成做T回合', String(stats.round_trips)],
        ['买入持有期末净资产', money(stats.benchmark_nav)], ['交易费用合计', money(stats.fees)],
        ['成交笔数', String(stats.fills)], ['应收股息', money(stats.receivable ?? result.nav.at(-1)?.receivable ?? 0)],
      ].map(([label, value]) => <div key={label} className="min-w-0"><dt className="text-xs text-muted">{label}</dt><dd className="mt-1 break-words font-mono text-lg">{value}</dd></div>)}
    </dl>
    <p className="text-xs text-muted">应收股息计入净资产，到账前不可用作买入资金；送股在可上市日期前锁定。</p>
    {result.warnings.length > 0 && <div className="rounded-btn border border-warning/30 bg-warning/5 p-3 text-sm" role="status"><h3 className="font-medium">回测提示</h3><ul className="mt-1 list-disc pl-5">{result.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul></div>}
    <NavChart nav={result.nav} />
    <div className="overflow-x-auto"><table className="w-full min-w-[540px] text-left text-xs"><caption className="mb-2 text-left font-medium text-sm">模拟成交明细（北京时间）</caption><thead className="text-muted"><tr>{['成交时间', '方向', '股数', '成交价', '费用'].map(h => <th key={h} className="border-b border-border px-2 py-2">{h}</th>)}</tr></thead><tbody>{result.fills.map((fill, i) => <tr key={i} className="border-b border-border/50"><td className="px-2 py-2">{fill.ts || fill.date}</td><td className="px-2 py-2">{fill.side === 'buy' ? '买入' : '卖出'}</td><td className="px-2 py-2 font-mono">{fill.qty}</td><td className="px-2 py-2 font-mono">{money(fill.price)}</td><td className="px-2 py-2 font-mono">{money(fill.fee)}</td></tr>)}</tbody></table>{!result.fills.length && <p className="py-3 text-sm text-muted">本次回测没有成交。</p>}</div>
    {result.events.length > 0 && <details><summary className="cursor-pointer text-sm">策略事件与拒单记录（{result.events.length}）</summary><ul className="mt-2 space-y-2 text-xs text-secondary">{result.events.map((event, i) => <li key={i} className="break-words rounded-btn bg-elevated p-2">{[event.date, event.ts, event.message ?? event.reason ?? event.status ?? event.type].filter(Boolean).map(String).join(' · ')}</li>)}</ul></details>}
  </section>
}

export function TStrategy() {
  const status = useQuery({ queryKey: QK.tStrategyStatus, queryFn: api.tStrategyStatus })
  return <div className="flex min-h-0 flex-1 flex-col overflow-y-auto text-foreground">
    <PageHeader title="底仓做T" subtitle="本地原价分钟数据 · A股 · 纯模拟" right={<Link to="/paper" className="shrink-0 text-sm text-accent">查看模拟盘</Link>} />
    <div className="mx-auto w-full max-w-7xl space-y-5 p-4 sm:p-5">
      {status.isPending && <p role="status">正在读取本地数据范围和模拟账户…</p>}
      {status.error && <div role="alert" className="space-y-2 text-sm text-danger"><p>{errorText(status.error)}</p><button className={buttonClass} onClick={() => void status.refetch()}>重新加载</button></div>}
      {status.data && <Workspace status={status.data} />}
    </div>
  </div>
}
function Workspace({ status }: { status: TStrategyStatus }) {
  const [config, setConfig] = useState(() => seedConfig(status))
  const active = status.accounts.filter(a => a.status === 'active')
  const [account, setAccount] = useState(() => active.some(a => a.id === 'sim_t') ? 'sim_t' : active[0]?.id ?? 'sim_t')
  const [result, setResult] = useState<TStrategyResult | null>(null)
  const [error, setError] = useState('')
  const lock = useRef(false)
  const client = useQueryClient()
  const backtest = useMutation({ mutationFn: api.tStrategyBacktest, onSuccess: data => {
    setResult(data); client.setQueryData(QK.tStrategyRun(data.id), data)
    void client.invalidateQueries({ queryKey: QK.tStrategyStatus })
  } })
  const importPaper = useMutation({ mutationFn: api.tStrategyImportPaper, onSuccess: () => {
    void client.invalidateQueries({ queryKey: QK.paperAll }); void client.invalidateQueries({ queryKey: QK.tStrategyStatus })
  } })
  const step = useMutation({ mutationFn: api.tStrategyManualStep, onSuccess: () => {
    void client.invalidateQueries({ queryKey: QK.paperAll }); void client.invalidateQueries({ queryKey: QK.tStrategyStatus })
  } })
  const history = useMutation({ mutationFn: (id: string) => client.fetchQuery({ queryKey: QK.tStrategyRun(id), queryFn: () => api.tStrategyRun(id) }), onSuccess: setResult })
  const busy = backtest.isPending || importPaper.isPending || step.isPending || history.isPending
  async function perform(action: () => Promise<unknown>) {
    if (lock.current) return
    lock.current = true; setError('')
    try { await action() } catch (e) { setError(errorText(e)) } finally { lock.current = false }
  }
  function run(event: FormEvent) {
    event.preventDefault()
    if (lock.current) return
    const validation = validate(config, true)
    if (validation) { setError(validation); return }
    setResult(null); importPaper.reset()
    void perform(() => backtest.mutateAsync(config))
  }
  function evaluate() {
    const validation = validate(config, false)
    if (validation) { setError(validation); return }
    if (active.length > 0 && !active.some(a => a.id === account)) { setError('请先在模拟盘创建或选择一个启用中的账户'); return }
    step.reset(); void perform(() => step.mutateAsync({ ...config, account }))
  }
  function field<K extends keyof TStrategyConfig>(key: K, value: TStrategyConfig[K]) { setConfig(c => ({ ...c, [key]: value })) }
  const numberFields = [
    ['initial_cash', '初始资金（元）', '0.01'], ['base_qty', '底仓股数', '100'], ['trade_qty', '每次做T股数', '100'],
    ['rise_pct', '上涨阈值（%）', '0.1'], ['pullback_pct', '回落阈值（%）', '0.1'],
    ['commission_pct', '佣金比例（0.00025 = 万2.5）', '0.00001'], ['slippage_bps', '滑点（基点，1 = 0.01%）', '1'],
  ] as const
  return <>
    <section className="rounded-card border border-accent/20 bg-accent/5 p-4 text-sm leading-6 text-secondary">
      <p className="font-medium text-foreground">仅使用本机已保存的股票原价分钟线</p>
      <p>本地范围：{status.coverage.start && status.coverage.end ? `${status.coverage.start} 至 ${status.coverage.end}` : '暂无可用分钟数据，请先同步股票分钟线'} · {status.coverage.sessions} 个交易日 · {status.coverage.bars} 根。单次限一只股票、最多 90 个自然日；各标的数据完整性由服务器检查。</p>
      <p>先建立底仓，遵守 T+1；上涨触发卖出，回落或到时回补，每天最多一轮。完成分钟线产生信号后，仅在相邻下一分钟开盘模拟成交。包含佣金、印花税、过户费和滑点；涨跌停或缺失数据可能拒单，未完成回补会保留。</p>
    </section>
    <form className={panelClass} onSubmit={run} noValidate>
      <h2 className="font-semibold">策略参数</h2>
      <fieldset disabled={busy} className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <label className="space-y-1 text-xs text-secondary"><span>股票代码</span><input className={inputClass} name="symbol" value={config.symbol} onChange={e => field('symbol', e.target.value.trim().toUpperCase())} placeholder="000001.SZ" /></label>
        {(['start', 'end'] as const).map((key, i) => <label key={key} className="space-y-1 text-xs text-secondary"><span>{i === 0 ? '开始日期' : '结束日期'}</span><input className={inputClass} type="date" name={key} value={config[key]} onChange={e => field(key, e.target.value)} /></label>)}
        {numberFields.map(([key, label, step]) => <label key={key} className="space-y-1 text-xs text-secondary"><span>{label}</span><input className={inputClass} type="number" name={key} step={step} value={Number.isNaN(config[key]) ? '' : config[key]} onChange={e => field(key, e.target.value === '' ? NaN : Number(e.target.value))} /></label>)}
        {(['start_time', 'restore_time'] as const).map((key, i) => <label key={key} className="space-y-1 text-xs text-secondary"><span>{i === 0 ? '开始评估时间（北京）' : '回补时间（北京）'}</span><input className={inputClass} name={key} type="time" value={config[key]} onChange={e => field(key, e.target.value)} /></label>)}
      </fieldset>
      <button className={`${buttonClass} bg-accent/10 text-accent`} type="submit" disabled={busy}>{backtest.isPending ? '正在回测…' : '运行回测'}</button>
    </form>
    {error && <p role="alert" className="rounded-btn border border-danger/20 bg-danger/5 p-3 text-sm text-danger break-words">{error}</p>}
    {result && <>
      <ResultPanel result={result} />
      <div className="flex flex-wrap items-center gap-3">
        <button type="button" className={buttonClass} disabled={busy || !!importPaper.data} onClick={() => void perform(() => importPaper.mutateAsync(result.id))}>{importPaper.isPending ? '正在生成…' : '生成历史模拟账户'}</button>
        <p className="text-xs text-muted">从本次回测账本生成已冻结历史账户，只供查阅和对比。</p>
        {importPaper.data && <p role="status" className="text-sm text-accent">历史模拟账户 {importPaper.data.account} 已冻结。<Link to="/paper" className="underline">前往模拟盘查看</Link></p>}
      </div>
    </>}
    <section className={panelClass}>
      <h2 className="font-semibold">手动模拟评估</h2>
      <p className="text-sm text-secondary">使用上方股票、数量和阈值，回测日期不参与评估。仅评估服务器当前可用的已完成分钟线；每点击一次评估一次，同一分钟重复点击由服务器去重。信号挂单须等待之后可成交的分钟线；不会后台自动评估。</p>
      <div className="flex flex-wrap items-end gap-3">
        <label className="min-w-0 flex-1 space-y-1 text-xs text-secondary"><span>启用中的模拟账户</span><select className={inputClass} aria-label="模拟账户" value={account} disabled={busy} onChange={e => { setAccount(e.target.value); step.reset() }}>{!active.length && <option value="sim_t">新建做T模拟账户 · sim_t · 初始资金 {money(config.initial_cash)}</option>}{active.map(a => <option key={a.id} value={a.id}>{a.name || a.id} · {a.id} · 现金 {money(a.cash)}</option>)}</select></label>
        <button className={buttonClass} disabled={busy || (active.length > 0 && !active.some(a => a.id === account))} onClick={evaluate}>{step.isPending ? '正在评估…' : '执行一次模拟评估'}</button>
      </div>
      <p className="text-xs text-muted">{!active.length ? '点击评估后，服务器仅在数据就绪时新建 sim_t，使用上方初始资金和费率；新买入底仓须等下个交易日才可卖出。' : '已有账户保留实际余额和费用规则；上方初始资金及费率仅用于回测和新建账户。'}</p>
      {step.data && <div role="status" aria-live="polite" className="space-y-1 rounded-btn border border-border bg-elevated p-3 text-sm"><p className="font-medium">{statusLabels[step.data.status] || '评估结果'} · {step.data.account}</p><p>{step.data.message}</p>{step.data.order && <p className="text-xs text-muted">已记录模拟订单，订单详情请到模拟盘查看。</p>}{!!step.data.filled?.length && <p>本次新增模拟成交 {step.data.filled.length} 笔，详情请到模拟盘查看。</p>}</div>}
    </section>
    {!!status.runs.length && <section className={panelClass}><h2 className="font-semibold">历史回测</h2><ul className="space-y-2">{status.runs.map(run => <li key={run.id}><button className={`${buttonClass} text-left`} disabled={busy} onClick={() => { importPaper.reset(); void perform(() => history.mutateAsync(run.id)) }}>{run.params.symbol} · {run.params.start} 至 {run.params.end} · 收益 {run.stats.return_pct.toFixed(2)}% · 查看 {run.id}</button></li>)}</ul></section>}
  </>
}
