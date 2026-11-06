import { lazy } from 'react'
import { ArrowLeftRight } from 'lucide-react'
import type { FrontendExtension } from '@/extensions/types'

const TStrategy = lazy(() => import('./Page').then(m => ({ default: m.TStrategy })))
const extension: FrontendExtension = {
  id: 'custom.t-strategy',
  apiVersion: 1,
  routes: [{ id: 't-strategy', path: '/t-strategy', component: TStrategy }],
  navigation: [{ id: 't-strategy', routeId: 't-strategy', label: '底仓做T', icon: ArrowLeftRight }],
}
export default extension
