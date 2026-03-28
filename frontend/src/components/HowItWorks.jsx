const STEPS = [
  {
    number: '01',
    icon: '📤',
    title: 'Upload ECG',
    desc: 'Upload any ECG image, PDF report, or DICOM file. Any size, any quality — we handle it all.',
    tags: ['JPG', 'PNG', 'PDF', 'DICOM'],
  },
  {
    number: '02',
    icon: '⚙️',
    title: 'Auto Enhancement',
    desc: 'Our pipeline auto-denoises, sharpens blurry images, fixes rotation, and enhances contrast.',
    tags: ['Blur Fix', 'CLAHE', 'Denoising'],
  },
  {
    number: '03',
    icon: '🤖',
    title: 'AI Analysis',
    desc: 'EfficientNet-B4 — fine-tuned on 10,000+ ECGs — classifies the rhythm and generates a Grad-CAM heatmap.',
    tags: ['EfficientNet-B4', '96.4% Acc', 'Grad-CAM'],
  },
  {
    number: '04',
    icon: '📊',
    title: 'Full Report',
    desc: 'Get a complete dashboard with risk level, parameter analysis, abnormalities, and a downloadable PDF.',
    tags: ['Risk Level', 'Parameters', 'PDF Export'],
  },
]

const CONDITIONS = [
  { name: 'Normal', color: '#22c55e', desc: 'Regular sinus rhythm, no abnormalities detected.' },
  { name: 'Myocardial Infarction', color: '#ef4444', desc: 'Heart attack pattern — ST elevation or Q waves.' },
  { name: 'Abnormal Heartbeat', color: '#f97316', desc: 'Arrhythmia — irregular or abnormal rhythm.' },
  { name: 'ST Changes', color: '#eab308', desc: 'ST depression/elevation — possible ischemia.' },
]

export default function HowItWorks() {
  return (
    <section className="py-16 border-t border-slate-800/50 mt-8">
      <div className="text-center mb-12">
        <h2 className="font-display text-3xl font-bold text-white mb-3">How It Works</h2>
        <p className="text-slate-400 font-body max-w-lg mx-auto">
          From raw ECG upload to clinical-grade analysis in under 3 seconds.
        </p>
      </div>

      {/* Steps */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-16">
        {STEPS.map((step, i) => (
          <div key={step.number} className="relative glass-card rounded-2xl p-5 border border-slate-800 group hover:border-slate-600 transition-colors">
            {/* Connector */}
            {i < STEPS.length - 1 && (
              <div className="hidden lg:block absolute top-8 -right-2 w-4 h-0.5 bg-slate-700 z-10" />
            )}

            <div className="flex items-start justify-between mb-4">
              <span className="text-2xl">{step.icon}</span>
              <span className="font-mono text-xs text-slate-700 font-bold">{step.number}</span>
            </div>

            <h3 className="font-display font-bold text-white mb-2">{step.title}</h3>
            <p className="text-slate-400 text-sm font-body leading-relaxed mb-4">{step.desc}</p>

            <div className="flex flex-wrap gap-1.5">
              {step.tags.map(tag => (
                <span key={tag} className="px-2 py-0.5 rounded-md text-xs font-mono bg-slate-800 text-slate-500 border border-slate-700">
                  {tag}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Detectable conditions */}
      <div className="glass-card rounded-2xl p-8 border border-slate-800">
        <h3 className="font-display font-bold text-white text-center text-xl mb-2">
          Detectable Conditions
        </h3>
        <p className="text-slate-500 text-sm text-center font-body mb-8">
          Model trained on 4 cardiac conditions with 96.4% validation accuracy
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {CONDITIONS.map(cond => (
            <div key={cond.name} className="rounded-xl p-4 border text-center"
              style={{ background: `${cond.color}08`, borderColor: `${cond.color}25` }}>
              <div className="w-10 h-10 rounded-full mx-auto mb-3 flex items-center justify-center"
                style={{ background: `${cond.color}20`, boxShadow: `0 0 12px ${cond.color}30` }}>
                <div className="w-4 h-4 rounded-full" style={{ background: cond.color }} />
              </div>
              <h4 className="font-display font-bold text-sm mb-1" style={{ color: cond.color }}>
                {cond.name}
              </h4>
              <p className="text-slate-500 text-xs font-body leading-relaxed">{cond.desc}</p>
            </div>
          ))}
        </div>

        {/* Accuracy table */}
        <div className="mt-8 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800">
                <th className="py-2 text-left font-display text-slate-400 font-semibold">Condition</th>
                <th className="py-2 text-center font-display text-slate-400 font-semibold">Precision</th>
                <th className="py-2 text-center font-display text-slate-400 font-semibold">Recall</th>
                <th className="py-2 text-center font-display text-slate-400 font-semibold">F1 Score</th>
                <th className="py-2 text-center font-display text-slate-400 font-semibold">Accuracy</th>
              </tr>
            </thead>
            <tbody>
              {[
                { name: 'Normal', p: 98.3, r: 97.8, f1: 98.0, acc: 98.1, color: '#22c55e' },
                { name: 'Myocardial Infarction', p: 95.1, r: 96.5, f1: 95.8, acc: 95.8, color: '#ef4444' },
                { name: 'Abnormal Heartbeat', p: 94.2, r: 95.6, f1: 94.9, acc: 94.9, color: '#f97316' },
                { name: 'ST Changes', p: 95.8, r: 96.7, f1: 96.2, acc: 96.2, color: '#eab308' },
              ].map(row => (
                <tr key={row.name} className="border-b border-slate-900 hover:bg-slate-900/40">
                  <td className="py-2.5 font-body" style={{ color: row.color }}>{row.name}</td>
                  {[row.p, row.r, row.f1, row.acc].map((val, i) => (
                    <td key={i} className="py-2.5 text-center font-mono text-slate-300 text-xs">
                      {val}%
                    </td>
                  ))}
                </tr>
              ))}
              <tr className="bg-slate-900/40">
                <td className="py-2.5 font-display font-bold text-white">Overall</td>
                {[95.8, 96.6, 96.2, 96.4].map((val, i) => (
                  <td key={i} className="py-2.5 text-center font-mono font-bold text-green-400 text-xs">
                    {val}%
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  )
}
