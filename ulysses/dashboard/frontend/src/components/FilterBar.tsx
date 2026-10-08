import type { JobFilters } from '../api'

const STATUSES = ['new', 'notified', 'drafted', 'built', 'skipped', 'archived', 'won', 'lost']
const CATEGORIES = ['tier1', 'tier2', 'tier3']

export function FilterBar({
  filters,
  onChange,
}: {
  filters: JobFilters
  onChange: (filters: JobFilters) => void
}) {
  return (
    <div className="filter-bar">
      <select
        value={filters.status ?? ''}
        onChange={(e) => onChange({ ...filters, status: e.target.value || undefined })}
      >
        <option value="">All statuses</option>
        {STATUSES.map((status) => (
          <option key={status} value={status}>
            {status}
          </option>
        ))}
      </select>

      <select
        value={filters.category ?? ''}
        onChange={(e) => onChange({ ...filters, category: e.target.value || undefined })}
      >
        <option value="">All tiers</option>
        {CATEGORIES.map((category) => (
          <option key={category} value={category}>
            {category}
          </option>
        ))}
      </select>

      <select
        value={filters.source ?? ''}
        onChange={(e) =>
          onChange({ ...filters, source: (e.target.value || undefined) as JobFilters['source'] })
        }
      >
        <option value="">Email + manual</option>
        <option value="email">Email only</option>
        <option value="manual">Manual only</option>
      </select>

      <label className="filter-bar__number">
        Min score
        <input
          type="number"
          min={0}
          max={100}
          value={filters.minScore ?? ''}
          placeholder="0"
          onChange={(e) =>
            onChange({
              ...filters,
              minScore: e.target.value === '' ? undefined : Number(e.target.value),
            })
          }
        />
      </label>

      <label className="filter-bar__toggle">
        <input
          type="checkbox"
          checked={filters.maxProposals === 5}
          onChange={(e) => onChange({ ...filters, maxProposals: e.target.checked ? 5 : undefined })}
        />
        Low competition (≤5 proposals)
      </label>

      <label className="filter-bar__toggle">
        <input
          type="checkbox"
          checked={filters.hasRedFlags === true}
          onChange={(e) => onChange({ ...filters, hasRedFlags: e.target.checked || undefined })}
        />
        Has red flags
      </label>

      {(filters.status ||
        filters.category ||
        filters.source ||
        filters.minScore ||
        filters.maxProposals ||
        filters.hasRedFlags) && (
        <button className="filter-bar__clear" onClick={() => onChange({})}>
          Clear filters
        </button>
      )}
    </div>
  )
}
