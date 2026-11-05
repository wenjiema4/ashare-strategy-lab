// @vitest-environment jsdom
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { startBacktest } from './backtestTask'
import { startRobust } from './robustTask'
import { startWalkForward } from './walkforwardTask'

let requests: string[] = []
class CapturingEventSource {
  constructor(url: string) { requests.push(url) }
  addEventListener() {}
  close() {}
}

beforeEach(() => {
  requests = []
  localStorage.clear()
  vi.stubGlobal('EventSource', CapturingEventSource)
  vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, json: async () => ({ items: [] }) })))
})
afterEach(() => vi.unstubAllGlobals())

const fees = { fee_model: 'ashare_v1' as const, commission_pct: 0.00025,
  stamp_tax_pct: null, slippage_bps: 5 }

it('preserves monetary cost model in ordinary backtest requests', async () => {
  startBacktest({ strategy_id: 'demo', ...fees })
  await vi.waitFor(() => expect(requests.length).toBe(1))
  const query = new URL(requests[0], 'http://localhost').searchParams
  expect(query.get('fee_model')).toBe('ashare_v1')
  expect(query.get('commission_pct')).toBe('0.00025')
  expect(query.has('stamp_tax_pct')).toBe(false)
})

it('preserves explicit zero stamp tax in robust optimization', () => {
  startRobust({ strategy_id: 'demo', param_grid: { n: [5] }, objective: 'sortino',
    ...fees, stamp_tax_pct: 0 })
  const query = new URL(requests[0], 'http://localhost').searchParams
  expect(query.get('fee_model')).toBe('ashare_v1')
  expect(query.get('stamp_tax_pct')).toBe('0')
})

it('uses the same cost settings in walk-forward validation', () => {
  startWalkForward({ strategy_id: 'demo', param_grid: { n: [5] }, objective: 'sortino',
    train_days: 252, test_days: 63, step_days: 63, ...fees })
  const query = new URL(requests[0], 'http://localhost').searchParams
  expect(query.get('fee_model')).toBe('ashare_v1')
  expect(query.get('commission_pct')).toBe('0.00025')
  expect(query.get('slippage_bps')).toBe('5')
  expect(query.has('stamp_tax_pct')).toBe(false)
})
