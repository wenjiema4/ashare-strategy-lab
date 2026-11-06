// @vitest-environment jsdom
import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { TStrategy } from './Page'
import { QK } from '@/lib/queryKeys'

const config = { symbol: '000001.SZ', start: '2026-10-01', end: '2026-10-31', initial_cash: 100000, base_qty: 600, trade_qty: 300, rise_pct: 1, pullback_pct: 1, start_time: '09:45', restore_time: '14:55', commission_pct: 0.00025, slippage_bps: 5 }
const result = { id: 'run_1', params: config, stats: { initial_cash: 100000, final_nav: 101000, return_pct: 1, benchmark_nav: 100500, excess_pnl: 500, round_trips: 2, fills: 5, fees: 33, receivable: 100 }, nav: [{ date: '2026-10-09', cash: 95000, mv: 6000, nav: 101000, receivable: 100 }], fills: [{ date: '2026-10-09', ts: '2026-10-09 09:46:00', side: 'sell', qty: 300, price: 10.8, fee: 6, kind: 't_sell' }], events: [{ type: 'rejected', message: '跌停无法成交' }], warnings: ['尚有回补义务'], coverage: { start: '2026-10-09', end: '2026-10-09', bars: 240, sessions: 1 } }
const status = { defaults: config, coverage: { start: '2026-10-08', end: '2026-10-09', bars: 480, sessions: 2 }, runs: [], accounts: [{ id: 'frozen', name: '历史账户', status: 'frozen', cash: 1 }, { id: 'sim_t', name: '底仓模拟', status: 'active', cash: 100000 }] }
let root: Root
let client: QueryClient
let node: HTMLDivElement
let requests: { path: string; body: any }[]
let reply: (path: string, body: any) => Response | Promise<Response>
const json = (value: unknown, code = 200) => new Response(JSON.stringify(value), { status: code, headers: { 'Content-Type': 'application/json' } })
async function wait(assert: () => void) { await vi.waitFor(async () => { await act(async () => { await new Promise(r => setTimeout(r, 0)) }); assert() }) }
async function render() {
  root = createRoot(node)
  await act(async () => { root.render(<MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><QueryClientProvider client={client}><TStrategy /></QueryClientProvider></MemoryRouter>) })
  await wait(() => expect(node.querySelector('input[name="start"]')).not.toBeNull())
}
const button = (text: string) => [...node.querySelectorAll('button')].find(b => b.textContent === text)!
async function click(text: string) { await act(async () => button(text).click()) }
async function input(name: string, value: string) {
  const el = node.querySelector<HTMLInputElement>(`input[name="${name}"]`)!
  await act(async () => { Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(el, value); el.dispatchEvent(new Event('input', { bubbles: true })) })
}
beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  node = document.createElement('div'); document.body.append(node)
  client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  requests = []
  reply = path => json(path.endsWith('/status') ? status : path.endsWith('/paper') ? { account: 't_history_1', frozen: true } : path.endsWith('/step') ? { status: 'stale_data', message: '本地分钟数据已过期，请先同步当日数据', account: 'sim_t' } : result)
  vi.stubGlobal('fetch', vi.fn(async (url, init) => { const path = String(url); const body = init?.body ? JSON.parse(init.body) : undefined; requests.push({ path, body }); return reply(path, body) }))
})
afterEach(() => { act(() => root?.unmount()); client.clear(); node.remove(); vi.unstubAllGlobals() })

it('seeds coverage and active account without evaluating or backtesting on page load', async () => {
  await render()
  expect(node.querySelector<HTMLInputElement>('input[name="start"]')!.value).toBe('2026-10-08')
  expect(node.querySelector<HTMLInputElement>('input[name="end"]')!.value).toBe('2026-10-09')
  expect(node.querySelector<HTMLInputElement>('input[name="symbol"]')!.value).toBe('000001.SZ')
  expect([...node.querySelectorAll('select[aria-label="模拟账户"] option')].map(o => o.getAttribute('value'))).toEqual(['sim_t'])
  expect(requests.map(r => r.path)).toEqual(['/api/t-strategy/status'])
  expect(button('生成历史模拟账户')).toBeUndefined()
})
it('runs edited configuration and shows results, warnings, receivables and frozen import', async () => {
  await render(); await input('rise_pct', '2'); await click('运行回测')
  await wait(() => expect(node.textContent).toContain('101,000.00'))
  expect(requests.find(r => r.path.endsWith('/backtest'))!.body.rise_pct).toBe(2)
  expect(node.textContent).toContain('1.00%')
  expect(node.textContent).toContain('尚有回补义务')
  expect(node.textContent).toContain('应收股息')
  expect(node.textContent).toContain('100.00')
  expect(node.textContent).toContain('跌停无法成交')
  expect(node.querySelector('svg[aria-label="账户净值曲线"]')).not.toBeNull()
  expect(node.querySelector('table')?.textContent).toContain('300')
  const invalidate = vi.spyOn(client, 'invalidateQueries')
  await click('生成历史模拟账户')
  await wait(() => expect(node.textContent).toContain('t_history_1'))
  expect(requests.find(r => r.path.endsWith('/paper'))!.body).toEqual({})
  expect(node.textContent).toContain('已冻结')
  expect(invalidate).toHaveBeenCalledWith({ queryKey: QK.paperAll })
  expect(node.querySelector('a[href="/paper"]')).not.toBeNull()
})
it('shows backtest rejection and hides import for an earlier result after failed retry', async () => {
  await render(); await click('运行回测'); await wait(() => expect(button('生成历史模拟账户')).toBeDefined())
  reply = path => path.endsWith('/backtest') ? json({ detail: '分钟数据不完整，请先补齐' }, 422) : json(status)
  await click('运行回测')
  await wait(() => expect(node.querySelector('[role="alert"]')?.textContent).toContain('分钟数据不完整'))
  expect(button('生成历史模拟账户')).toBeUndefined()
})
it('evaluates only after clicking and displays stale and pending statuses without invented fills', async () => {
  await render(); await click('执行一次模拟评估')
  await wait(() => expect(node.textContent).toContain('本地分钟数据已过期'))
  expect(requests.find(r => r.path.endsWith('/step'))!.body).toEqual({ account: 'sim_t', ...config, start: '2026-10-08', end: '2026-10-09' })
  reply = path => json(path.endsWith('/status') ? status : { status: 'pending', message: '信号已记录，等待下一根可成交分钟线', account: 'sim_t', order: { id: 'order_1' } })
  await click('执行一次模拟评估')
  await wait(() => expect(node.textContent).toContain('等待下一根可成交分钟线'))
  expect(node.textContent).toContain('待成交')
})
it('blocks invalid board lots and excessive trade quantity before requests', async () => {
  await render(); await input('trade_qty', '700'); await click('运行回测')
  expect(node.textContent).toContain('每次做T股数不能超过底仓')
  await input('trade_qty', '250'); await click('运行回测')
  expect(node.textContent).toContain('100 股的整数倍')
  expect(requests.some(r => r.path.endsWith('/backtest'))).toBe(false)
})
it('locks mutations during a pending request and restores controls after its error', async () => {
  let reject!: (e: Error) => void
  reply = path => path.endsWith('/backtest') ? new Promise((_resolve, fail) => { reject = fail }) : json(status)
  await render(); await click('运行回测')
  await wait(() => expect(button('正在回测…')?.disabled).toBe(true))
  expect(button('执行一次模拟评估').disabled).toBe(true)
  await act(async () => reject(new Error('网络中断，请重试')))
  await wait(() => expect(node.querySelector('[role="alert"]')?.textContent).toContain('网络中断'))
  expect(button('运行回测').disabled).toBe(false)
})

it('rejects out-of-bounds strategy thresholds and slippage before backtest submission', async () => {
  await render(); await input('rise_pct', '31'); await click('运行回测')
  expect(node.textContent).toContain('30%')
  expect(requests.some(r => r.path.endsWith('/backtest'))).toBe(false)
  await input('rise_pct', '1'); await input('slippage_bps', '101'); await click('运行回测')
  expect(node.textContent).toContain('100 个基点')
  expect(requests.some(r => r.path.endsWith('/backtest'))).toBe(false)
})
it('loads a saved run through the shared client and uses its result for frozen import', async () => {
  reply = path => json(path.endsWith('/status') ? { ...status, runs: [{ id: result.id, params: config, stats: result.stats }] } : result)
  await render()
  await act(async () => [...node.querySelectorAll('button')].find(b => b.textContent?.includes('查看 run_1'))!.click())
  await wait(() => expect(node.textContent).toContain('101,000.00'))
  expect(requests.some(r => r.path === '/api/t-strategy/runs/run_1')).toBe(true)
  expect(button('生成历史模拟账户').disabled).toBe(false)
})
it('offers explicit sim_t creation when no active accounts exist without creating on load', async () => {
  reply = path => json(path.endsWith('/status') ? { ...status, accounts: [status.accounts[0]] } : { status: 'stale_data', message: '分钟数据尚未就绪，未创建账户', account: 'sim_t' })
  await render()
  expect(button('执行一次模拟评估').disabled).toBe(false)
  expect(node.textContent).toContain('新建做T模拟账户')
  expect(requests.some(r => r.path.endsWith('/step'))).toBe(false)
  await input('initial_cash', '123456'); await click('执行一次模拟评估')
  await wait(() => expect(node.textContent).toContain('未创建账户'))
  expect(requests.find(r => r.path.endsWith('/step'))!.body).toMatchObject({ account: 'sim_t', initial_cash: 123456 })
})
it('shows initial status errors with a retry action without posting mutations', async () => {
  reply = () => json({ detail: '本地状态读取失败' }, 503)
  root = createRoot(node)
  await act(async () => root.render(<MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><QueryClientProvider client={client}><TStrategy /></QueryClientProvider></MemoryRouter>))
  await wait(() => expect(node.querySelector('[role="alert"]')?.textContent).toContain('本地状态读取失败'))
  expect(button('重新加载')).toBeDefined()
  expect(requests.every(r => r.body === undefined)).toBe(true)
})
it('keeps failed frozen import retryable and displays the API rejection', async () => {
  await render(); await click('运行回测'); await wait(() => expect(button('生成历史模拟账户')).toBeDefined())
  reply = path => path.endsWith('/paper') ? json({ detail: '历史账户账本不可用' }, 409) : json(status)
  await click('生成历史模拟账户')
  await wait(() => expect(node.querySelector('[role="alert"]')?.textContent).toContain('历史账户账本不可用'))
  expect(button('生成历史模拟账户').disabled).toBe(false)
  expect(node.textContent).not.toContain('已冻结。')
})
it('clears previous manual status and exposes API failure for an explicit retry', async () => {
  await render(); await click('执行一次模拟评估'); await wait(() => expect(node.textContent).toContain('本地分钟数据已过期'))
  reply = path => path.endsWith('/step') ? json({ detail: '账户费率不一致，请修改参数' }, 409) : json(status)
  await click('执行一次模拟评估')
  await wait(() => expect(node.querySelector('[role="alert"]')?.textContent).toContain('账户费率不一致'))
  expect(node.textContent).not.toContain('本地分钟数据已过期')
  expect(button('执行一次模拟评估').disabled).toBe(false)
})
