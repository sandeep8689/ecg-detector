import { useEffect, useState } from 'react'

export default function AbnormalityMeter({ count, total, percentage, riskLevel, abnormalities }) {
  const [animated, setAnimated] = useState(false)

  useEffect(() => {
    const t = setTimeout(() => setAnimated(true), 200)
    return () => clearTimeout(t)
  }, [])

  const color = riskLevel === 'CRITICAL' ? '#ef4444' : riskLevel === 'HIGH' ? '#f97316' : '#22c55e'
  const bgColor = riskLevel === 'CRITICAL' ? 'bg-red-950/40 border-red-800/30' :
                  riskLevel === 'HIGH' ? 'bg-orange-950/40 border-orange-800/30' :
                  'bg-green-950/40 border-green-800/30'

  // Radial progress arc
  const radius = 54
  const circumference = 2 * Math.PI * radius
  const strokeDash = animated ? (percentage / 100) * circumference : 0

  return (
    <div className={`glass-card rounded-2xl p-6 mb-6 border ${bgColor}`}>
      <h3 className="font-display text-sm font-semibold text-slate-300 uppercase tracking-widest mb-5">
        Abnormality Assessment
      </h3>

      <div className="flex flex-col md:flex-row items-center gap-8">
        {/* Radial meter */}
        <div className="relative flex-shrink-0">
          <svg width="140" height="140" viewBox="0 0 140 140">
            {/* Background track */}
            <circle cx="70" cy="70" r={radius} fill="none" stroke="#1e2d45" strokeWidth="10" />
            {/* Tick marks */}
            {Array.from({ length: total }).map((_, i) => {
              const angle = (i / total) * 360 - 90
              const rad = angle * (Math.PI / 180)
              const x1 = 70 + (radius - 7) * Math.cos(rad)
              const y1 = 70 + (radius - 7) * Math.sin(rad)
              const x2 = 70 + (radius + 7) * Math.cos(rad)
              const y2 = 70 + (radius + 7) * Math.sin(rad)
              const isAbnormal = i < count
              return (
                <line key={i} x1={x1} y1={y1} x2={x2} y2={y2}
                  stroke={isAbnormal ? color : '#1e3a5f'}
                  strokeWidth="3" strokeLinecap="round"
                  style={{ transition: `stroke 0.5s ease ${i * 0.08}s` }}
                />
              )
            })}
            {/* Progress arc */}
            <circle
              cx="70" cy="70" r={radius}
              fill="none"
              stroke={color}
              strokeWidth="6"
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={circumference - strokeDash}
              transform="rotate(-90 70 70)"
              style={{
                transition: 'stroke-dashoffset 1.2s cubic-bezier(0.4, 0, 0.2, 1)',
                filter: `drop-shadow(0 0 8px ${color}80)`
              }}
            />
            {/* Center text */}
            <text x="70" y="62" textAnchor="middle" fill="white"
              fontSize="26" fontWeight="700" fontFamily="Syne, sans-serif">
              {count}
            </text>
            <text x="70" y="78" textAnchor="middle" fill="#64748b"
              fontSize="11" fontFamily="DM Sans, sans-serif">
              of {total}
            </text>
            <text x="70" y="96" textAnchor="middle"
              fill={color} fontSize="13" fontWeight="600" fontFamily="Syne, sans-serif">
              {percentage}%
            </text>
          </svg>
          <p className="text-center text-xs text-slate-500 font-body mt-1">Abnormality Rate</p>
        </div>

        {/* Info */}
        <div className="flex-1 w-full">
          {/* Progress bar */}
          <div className="mb-4">
            <div className="flex justify-between text-xs font-body text-slate-400 mb-2">
              <span>Checks passed: {total - count}/{total}</span>
              <span style={{ color }}>{percentage}% abnormal</span>
            </div>
            <div className="h-3 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-1000 ease-out"
                style={{
                  width: animated ? `${percentage}%` : '0%',
                  background: `linear-gradient(90deg, ${color}80, ${color})`,
                  boxShadow: `0 0 12px ${color}60`
                }}
              />
            </div>
          </div>

          {/* Individual checks grid */}
          <div className="grid grid-cols-4 gap-2 mb-4">
            {Array.from({ length: total }).map((_, i) => {
              const isAbnormal = i < count
              return (
                <div
                  key={i}
                  className={`h-8 rounded-lg border flex items-center justify-center text-xs font-mono transition-all duration-500`}
                  style={{
                    transitionDelay: `${i * 80}ms`,
                    background: isAbnormal ? `${color}20` : '#0f1729',
                    borderColor: isAbnormal ? `${color}60` : '#1e3a5f',
                    color: isAbnormal ? color : '#475569',
                    boxShadow: isAbnormal ? `0 0 8px ${color}30` : 'none'
                  }}
                >
                  {isAbnormal ? '✗' : '✓'}
                </div>
              )
            })}
          </div>

          {/* Detected abnormalities list */}
          {abnormalities.length > 0 ? (
            <div>
              <p className="text-xs text-slate-500 font-body mb-2 uppercase tracking-wider">Detected:</p>
              <div className="flex flex-wrap gap-2">
                {abnormalities.map(abn => (
                  <span
                    key={abn}
                    className="px-2.5 py-1 rounded-full text-xs font-body border"
                    style={{
                      background: `${color}15`,
                      borderColor: `${color}40`,
                      color
                    }}
                  >
                    {abn}
                  </span>
                ))}
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-2 text-green-400 text-sm font-body">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              No significant abnormalities detected
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
