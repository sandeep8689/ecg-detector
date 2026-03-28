export default function ClinicalInfo({ info, disease }) {
  const sections = [
    {
      title: '📋 What This Means',
      content: info.what_it_means,
      type: 'text',
    },
    {
      title: '🩺 Possible Symptoms',
      content: info.symptoms,
      type: 'list',
      color: 'text-blue-300',
    },
    {
      title: '⏰ Recommended Action',
      content: info.urgency,
      type: 'highlight',
    },
    {
      title: '💡 Lifestyle Recommendations',
      content: info.lifestyle_tips,
      type: 'list',
      color: 'text-green-300',
    },
  ]

  const urgencyColor =
    info.urgency?.includes('EMERGENCY') || info.urgency?.includes('immediately')
      ? 'bg-red-950/60 border-red-700/40 text-red-300'
      : info.urgency?.includes('urgently') || info.urgency?.includes('soon')
      ? 'bg-orange-950/60 border-orange-700/40 text-orange-300'
      : 'bg-green-950/60 border-green-700/40 text-green-300'

  return (
    <div className="space-y-4">
      {/* Disease header card */}
      <div className="glass-card rounded-2xl p-6 border border-slate-800">
        <h3 className="font-display text-lg font-bold text-white mb-1">{disease}</h3>
        <p className="text-slate-300 font-body leading-relaxed">{info.short_summary}</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {sections.map(section => (
          <div key={section.title} className={`glass-card rounded-xl p-5 border border-slate-800
            ${section.type === 'highlight' ? `${urgencyColor} border` : ''}`}>
            <h4 className="font-display font-semibold text-white text-sm mb-3">{section.title}</h4>

            {section.type === 'text' && (
              <p className="text-slate-300 text-sm font-body leading-relaxed">{section.content}</p>
            )}

            {section.type === 'list' && (
              <ul className="space-y-2">
                {section.content?.map((item, i) => (
                  <li key={i} className={`flex items-start gap-2 text-sm font-body ${section.color || 'text-slate-300'}`}>
                    <span className="mt-0.5 text-xs opacity-60">◆</span>
                    {item}
                  </li>
                ))}
              </ul>
            )}

            {section.type === 'highlight' && (
              <p className="text-sm font-body font-medium leading-relaxed">{section.content}</p>
            )}
          </div>
        ))}
      </div>

      {/* Emergency contacts */}
      <div className="glass-card rounded-xl p-5 border border-red-900/20 bg-red-950/10">
        <h4 className="font-display font-semibold text-red-400 text-sm mb-3">🆘 Emergency Contacts (India)</h4>
        <div className="grid grid-cols-3 gap-3">
          {[
            { label: 'Ambulance', number: '108' },
            { label: 'National Health', number: '104' },
            { label: 'Emergency', number: '112' },
          ].map(({ label, number }) => (
            <div key={label} className="text-center">
              <p className="font-display text-2xl font-bold text-red-400">{number}</p>
              <p className="text-xs text-slate-500 font-body">{label}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
