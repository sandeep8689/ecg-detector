import { useEffect, useState } from 'react'

const STATUS_CONFIG = {
  NORMAL: { color: '#22c55e', bg: 'bg-green-950/40', border: 'border-green-800/30', label: 'Normal' },
  HIGH: { color: '#f97316', bg: 'bg-orange-950/40', border: 'border-orange-800/30', label: 'High' },
  LOW: { color: '#3b82f6', bg: 'bg-blue-950/40', border: 'border-blue-800/30', label: 'Low' },
  CRITICAL: { color: '#ef4444', bg: 'bg-red-950/40', border: 'border-red-800/30', label: 'Critical' },
}

function RangeBar({ value, min, max, status }) {
  const [animated, setAnimated] = useState(false)

  useEffect(() => {
    const t = setTimeout(() => setAnimated(true), 400)
    return () => clearTimeout(t)
  }, [])

  // Clamp position to 5%-95%
  const pct = Math.min(95, Math.max(5, ((value - min * 0.6) / (max * 1.4 - min * 0.6)) * 100))
  const color = STATUS_CONFIG[status]?.color || '#94a3b8'

  return (
    <div className="mt-3">
      {/* Labels */}
      <div className="flex justify-between text-xs font-mono text-slate-600 mb-1.5">
        <span>{min}</span>
        <span className="text-slate-400">Normal Range</span>
        <span>{max}</span>
      </div>

      {/* Bar */}
      <div className="relative h-2 rounded-full" style={{ background: '#1e2d45' }}>
        {/* Normal range highlight */}
        <div
          className="absolute top-0 bottom-0 rounded-full opacity-30"
          style={{
            left: '20%',
            width: '60%',
            background: '#22c55e',
          }}
        />
        {/* Value indicator */}
        <div
          className="absolute w-4 h-4 rounded-full border-2 border-slate-900 -top-1 transition-all duration-1000 ease-out"
          style={{
            left: animated ? `calc(${pct}% - 8px)` : 'calc(5% - 8px)',
            background: color,
            boxShadow: `0 0 8px ${color}80`,
          }}
        />
      </div>

      {/* Value pointer */}
      <div
        className="mt-1 transition-all duration-1000 ease-out flex justify-start"
        style={{ paddingLeft: animated ? `calc(${pct}% - 12px)` : '0%' }}
      >
        <svg width="24" height="8" viewBox="0 0 24 8">
          <polygon points="12,0 0,8 24,8" fill={color} opacity="0.7" />
        </svg>
      </div>
    </div>
  )
}

export default function RangesPanel({ ranges, riskLevel }) {
  const [expanded, setExpanded] = useState(null)
  const abnormalCount = ranges.filter(r => r.status !== 'NORMAL').length

  return (
    <div className="space-y-4">
      {/* Summary header */}
      <div className="glass-card rounded-2xl p-5 border border-slate-800">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-display font-semibold text-white">ECG Parameter Analysis</h3>
            <p className="text-slate-500 text-sm font-body mt-0.5">
              Comparing your ECG values against clinical normal ranges
            </p>
          </div>
          <div className="text-right">
            <p className="font-display text-2xl font-bold text-white">
              {ranges.length - abnormalCount}
              <span className="text-slate-600">/{ranges.length}</span>
            </p>
            <p className="text-xs text-slate-500 font-body">parameters normal</p>
          </div>
        </div>

        {/* Quick status strip */}
        <div className="flex gap-2 mt-4">
          {ranges.map(r => {
            const c = STATUS_CONFIG[r.status] || STATUS_CONFIG.NORMAL
            return (
              <div
                key={r.metric}
                className="flex-1 h-1.5 rounded-full"
                style={{ background: c.color, opacity: r.status === 'NORMAL' ? 0.3 : 1 }}
                title={`${r.metric}: ${r.status}`}
              />
            )
          })}
        </div>
      </div>

      {/* Individual range cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {ranges.map((range, i) => {
          const c = STATUS_CONFIG[range.status] || STATUS_CONFIG.NORMAL
          const isExpanded = expanded === range.metric
          const isAbnormal = range.status !== 'NORMAL'

          return (
            <div
              key={range.metric}
              className={`glass-card rounded-xl border cursor-pointer transition-all duration-200 hover:border-slate-600
                ${isAbnormal ? `${c.bg} ${c.border}` : 'border-slate-800'}
                ${isExpanded ? 'ring-1 ring-blue-500/30' : ''}
              `}
              onClick={() => setExpanded(isExpanded ? null : range.metric)}
            >
              <div className="p-4">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-2">
                    {isAbnormal && (
                      <div className="w-2 h-2 rounded-full animate-pulse" style={{ background: c.color }} />
                    )}
                    <span className="font-display font-semibold text-white text-sm">{range.metric}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span
                      className={`px-2 py-0.5 rounded-full text-xs font-display font-bold border`}
                      style={{
                        background: `${c.color}15`,
                        borderColor: `${c.color}40`,
                        color: c.color
                      }}
                    >
                      {c.label}
                    </span>
                    <svg
                      className={`w-4 h-4 text-slate-500 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                      fill="none" viewBox="0 0 24 24" stroke="currentColor"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </div>
                </div>

                {/* Main value display */}
                <div className="mt-3 flex items-end gap-2">
                  <span className="font-display text-3xl font-bold" style={{ color: c.color }}>
                    {range.value}
                  </span>
                  <span className="text-slate-500 text-sm font-mono mb-1">{range.unit}</span>
                  <span className="text-slate-600 text-xs font-body mb-1 ml-auto">
                    Normal: {range.normal_min}–{range.normal_max} {range.unit}
                  </span>
                </div>

                {/* Range bar */}
                <RangeBar
                  value={range.value}
                  min={range.normal_min}
                  max={range.normal_max}
                  status={range.status}
                />

                {/* Expanded description */}
                {isExpanded && (
                  <div className="mt-4 pt-4 border-t border-slate-700/50">
                    <p className="text-slate-400 text-xs font-body leading-relaxed">
                      {range.description}
                    </p>
                    {isAbnormal && (
                      <div className="mt-2 flex items-center gap-2 text-xs font-body"
                        style={{ color: c.color }}>
                        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.464 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z" />
                        </svg>
                        Value is {range.status === 'HIGH' ? 'above' : 'below'} normal range
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>

      <p className="text-xs text-slate-600 font-body text-center italic">
        Click any parameter card to expand details. Values are AI-estimated from ECG morphology.
      </p>
    </div>
  )
}
