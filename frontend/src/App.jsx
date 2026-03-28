import { useState } from 'react'
import './index.css'
import Header from './components/Header'
import UploadSection from './components/UploadSection'
import ResultDashboard from './components/ResultDashboard'
import HowItWorks from './components/HowItWorks'
import Footer from './components/Footer'

export default function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleReset = () => {
    setResult(null)
    setError(null)
  }

  return (
    <div className="min-h-screen grid-bg">
      {/* Ambient background blobs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
        <div className="absolute -top-40 -left-40 w-96 h-96 bg-red-900/10 rounded-full blur-3xl" />
        <div className="absolute top-1/3 -right-20 w-80 h-80 bg-blue-900/10 rounded-full blur-3xl" />
        <div className="absolute bottom-0 left-1/3 w-72 h-72 bg-red-900/5 rounded-full blur-3xl" />
      </div>

      <div className="relative z-10">
        <Header />

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-20">
          {!result ? (
            <>
              <UploadSection
                setResult={setResult}
                loading={loading}
                setLoading={setLoading}
                error={error}
                setError={setError}
              />
              <HowItWorks />
            </>
          ) : (
            <ResultDashboard result={result} onReset={handleReset} />
          )}
        </main>

        <Footer />
      </div>
    </div>
  )
}
