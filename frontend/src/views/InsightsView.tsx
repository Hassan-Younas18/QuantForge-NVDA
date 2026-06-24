import { api } from '../api/client'
import { useAsync } from '../hooks/useAsync'
import { ErrorBanner } from '../components/common/ErrorBanner'
import { LoadingSpinner } from '../components/common/LoadingSpinner'
import { InsightsPanel } from '../components/insights/InsightsPanel'

export function InsightsView() {
  const insights = useAsync(() => api.insights(), [])

  if (insights.loading) return <LoadingSpinner label="Generating insights..." />
  if (insights.error) return <ErrorBanner message={insights.error} onRetry={insights.refetch} />
  if (!insights.data) return null

  return <InsightsPanel insights={insights.data} />
}
