export default function RiskBadge({ riskLevel, size = 'md' }) {
  const config = {
    CRITICAL: {
      bg: 'bg-red-950/80',
      border: 'border-red-600/60',
      text: 'text-red-400',
      glow: 'risk-critical',
      icon: '🚨',
      label: 'CRITICAL',
      pulse: 'bg-red-500',
    },
    HIGH: {
      bg: 'bg-orange-950/80',
      border: 'border-orange-600/60',
      text: 'text-orange-400',
      glow: 'risk-high',
      icon: '⚠️',
      label: 'HIGH RISK',
      pulse: 'bg-orange-500',
    },
    LOW: {
      bg: 'bg-green-950/80',
      border: 'border-green-600/60',
      text: 'text-green-400',
      glow: 'risk-low',
      icon: '✅',
      label: 'NORMAL',
      pulse: 'bg-green-500',
    },
  }

  const c = config[riskLevel] || config.LOW

  const sizeClasses = {
    sm: 'px-2.5 py-1 text-xs gap-1.5',
    md: 'px-3 py-1.5 text-sm gap-2',
    lg: 'px-4 py-2.5 text-base gap-2.5',
  }

  return (
    <div className={`inline-flex items-center rounded-xl border font-display font-bold
      ${c.bg} ${c.border} ${c.text} ${c.glow} ${sizeClasses[size]}`}>
      <div className="relative flex-shrink-0">
        <div className={`absolute inset-0 rounded-full ${c.pulse} opacity-30 animate-ping`} />
        <div className={`relative w-2 h-2 rounded-full ${c.pulse}`} />
      </div>
      {c.icon} {c.label}
    </div>
  )
}
