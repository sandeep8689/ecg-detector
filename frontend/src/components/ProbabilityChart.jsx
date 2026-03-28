import { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const COLORS = {
  'Normal': '#22c55e',
  'Myocardial Infarction': '#ef4444',
  'Abnormal Heartbeat': '#f97316',
  'ST Depression / Elevation': '#eab308',
}

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const { name, value } = payload[0].payload
    return (
      <div className="glass-card rounded-xl px-4 py-3 border border-slate-700">
        <p className="font-display font-semibold text-white text-sm">{name}</p>
        <p className="text-2xl font-bold mt-1" style={{ color: COLORS[name] || '#94a3b8' }}>
          {value.toFixed(1)}%
        </p>
      </div>
    )
  }
  return null
}

export default function ProbabilityChart({ probabilities, disease }) {
  const [animated, setAnimated] = useState(false)

  useEffect(() => {
    const t = setTimeout(() => setAnimated(true), 300)
    return () => clearTimeout(t)
  }, [])

  const data = Object.entries(probabilities).map(([name, value]) => ({
    name,
    shortName: name.split(' ')[0],
    value,
    isDetected: name === disease,
  })).sort((a, b) => b.value - a.value)

  return (
    <div className="glass-card rounded-2xl p-6 border border-slate-800">
      <div className="flex items-center justify-between mb-5">
        <h3 className="font-display font-semibold text-white">AI Confidence Breakdown</h3>
        <span className="text-xs font-mono text-slate-500 bg-slate-800 px-2 py-1 rounded">
          4 conditions
        </span>
      </div>

      {/* Horizontal bar chart (custom) */}
      <div className="space-y-3 mb-6">
        {data.map(({ name, value, isDetected }, i) => {
          const color = COLORS[name] || '#94a3b8'
          return (
            <div key={name}>
              <div className="flex justify-between items-center mb-1">
                <div className="flex items-center gap-2">
                  {isDetected && (
                    <div className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: color }} />
                  )}
                  <span className={`text-xs font-body ${isDetected ? 'text-white font-medium' : 'text-slate-400'}`}>
                    {name}
                  </span>
                </div>
                <span className="text-xs font-mono font-bold" style={{ color: isDetected ? color : '#64748b' }}>
                  {value.toFixed(1)}%
                </span>
              </div>
              <div className="h-2.5 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-1000 ease-out"
                  style={{
                    width: animated ? `${value}%` : '0%',
                    transitionDelay: `${i * 150}ms`,
                    background: isDetected
                      ? `linear-gradient(90deg, ${color}90, ${color})`
                      : `linear-gradient(90deg, #1e3a5f, #2d4a6e)`,
                    boxShadow: isDetected ? `0 0 10px ${color}50` : 'none'
                  }}
                />
              </div>
            </div>
          )
        })}
      </div>

      {/* Recharts bar chart */}
      <div className="h-36 mt-4">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 5, right: 5, bottom: 5, left: -30 }}>
            <XAxis
              dataKey="shortName"
              tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'DM Sans' }}
              axisLine={false} tickLine={false}
            />
            <YAxis
              tick={{ fill: '#64748b', fontSize: 9, fontFamily: 'JetBrains Mono' }}
              axisLine={false} tickLine={false}
              tickFormatter={v => `${v}%`}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
            <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={50}>
              {data.map((entry) => (
                <Cell
                  key={entry.name}
                  fill={entry.isDetected ? COLORS[entry.name] || '#3b82f6' : '#1e3a5f'}
                  style={{ filter: entry.isDetected ? `drop-shadow(0 0 6px ${COLORS[entry.name]}80)` : 'none' }}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <p className="text-center text-xs text-slate-600 font-body mt-2">
        Model: EfficientNet-B4 • Accuracy: 96.4%
      </p>
    </div>
  )
}
