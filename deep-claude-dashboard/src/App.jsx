import { useState, useEffect, useCallback } from 'react'
import {
  Brain, Activity, Target, Eye, Zap, AlertTriangle, TrendingUp,
  BarChart3, Shield, MessageSquare, Sparkles, RefreshCw, ChevronDown, ChevronRight,
  Database, HeartPulse, Layers, Network, Search, Clock, Star
} from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  LineChart, Line, CartesianGrid, Legend, PieChart, Pie, Cell,
  ScatterChart, Scatter, ZAxis
} from 'recharts'

const API = 'http://localhost:8787/api'

function useFetch(endpoint) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const refetch = useCallback(() => {
    setLoading(true)
    fetch(`${API}${endpoint}`)
      .then(r => r.json())
      .then(d => { setData(d); setError(null) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [endpoint])

  useEffect(() => { refetch() }, [refetch])
  return { data, loading, error, refetch }
}

const COLORS = ['#3b82f6', '#8b5cf6', '#06b6d4', '#f59e0b', '#ef4444', '#22c55e', '#ec4899', '#f97316']

function StatCard({ icon: Icon, label, value, sub, color = 'blue' }) {
  const colorMap = {
    blue: 'from-blue-500/10 to-blue-600/5 border-blue-500/20',
    green: 'from-green-500/10 to-green-600/5 border-green-500/20',
    red: 'from-red-500/10 to-red-600/5 border-red-500/20',
    purple: 'from-purple-500/10 to-purple-600/5 border-purple-500/20',
    amber: 'from-amber-500/10 to-amber-600/5 border-amber-500/20',
    cyan: 'from-cyan-500/10 to-cyan-600/5 border-cyan-500/20',
  }
  const iconColor = {
    blue: 'text-blue-400', green: 'text-green-400', red: 'text-red-400',
    purple: 'text-purple-400', amber: 'text-amber-400', cyan: 'text-cyan-400',
  }
  return (
    <div className={`bg-gradient-to-br ${colorMap[color]} border rounded-xl p-4`}>
      <div className="flex items-center gap-2 mb-1">
        <Icon size={16} className={iconColor[color]} />
        <span className="text-xs text-gray-400 uppercase tracking-wider">{label}</span>
      </div>
      <div className="text-2xl font-bold text-gray-100">{value}</div>
      {sub && <div className="text-xs text-gray-500 mt-1">{sub}</div>}
    </div>
  )
}

function Panel({ title, icon: Icon, children, className = '', collapsible = false }) {
  const [open, setOpen] = useState(true)
  return (
    <div className={`bg-gray-900/80 border border-gray-800 rounded-xl overflow-hidden ${className}`}>
      <div
        className={`flex items-center gap-2 px-5 py-3 border-b border-gray-800 ${collapsible ? 'cursor-pointer hover:bg-gray-800/50' : ''}`}
        onClick={() => collapsible && setOpen(!open)}
      >
        {Icon && <Icon size={16} className="text-blue-400" />}
        <h3 className="text-sm font-semibold text-gray-200 flex-1">{title}</h3>
        {collapsible && (open ? <ChevronDown size={14} className="text-gray-500" /> : <ChevronRight size={14} className="text-gray-500" />)}
      </div>
      {open && <div className="p-5">{children}</div>}
    </div>
  )
}

// === OVERVIEW TAB ===
function OverviewTab() {
  const { data: overview, loading } = useFetch('/overview')
  const { data: domains } = useFetch('/domains')
  const { data: calibration } = useFetch('/calibration')

  if (loading || !overview) return <Loading />

  const calData = (calibration || []).map(c => ({
    domain: c.domain?.length > 12 ? c.domain.slice(0, 12) + '..' : c.domain,
    felt: c.avg_felt_confidence,
    actual: c.accuracy_ratio,
    error: c.avg_calibration_error,
    claims: c.total_claims,
  }))

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
        <StatCard icon={Brain} label="Memory Atoms" value={overview.atoms?.toLocaleString()} color="blue" />
        <StatCard icon={Target} label="Claims" value={overview.claims} sub={`${overview.resolved} resolved`} color="purple" />
        <StatCard icon={Shield} label="Accuracy" value={`${(overview.accuracy * 100).toFixed(0)}%`} sub={`${overview.correct}/${overview.resolved}`} color="green" />
        <StatCard icon={AlertTriangle} label="Avg Cal Error" value={overview.avg_calibration_error?.toFixed(3)} color="amber" />
        <StatCard icon={Zap} label="Interventions" value={overview.interventions} color="cyan" />
        <StatCard icon={Eye} label="Confabulations" value={overview.confabulations} color="red" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Panel title="Calibration by Domain" icon={BarChart3}>
          {calData.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={calData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="domain" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} domain={[0, 1]} />
                <Tooltip contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }} />
                <Legend />
                <Bar dataKey="felt" name="Felt Confidence" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="actual" name="Actual Accuracy" fill="#22c55e" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <Empty msg="No calibration data yet" />}
        </Panel>

        <Panel title="Domain Cognitive Map" icon={Brain}>
          {domains && domains.length > 0 ? (
            <div className="space-y-2 max-h-[280px] overflow-y-auto">
              {domains.map((d, i) => (
                <div key={i} className="flex items-center gap-3 p-2 rounded-lg bg-gray-800/50 hover:bg-gray-800 transition">
                  <div className="w-2 h-2 rounded-full" style={{ background: COLORS[i % COLORS.length] }} />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-gray-200 truncate">{d.domain}</div>
                    <div className="text-xs text-gray-500">{d.total} claims, {d.failures} failures</div>
                  </div>
                  <div className="text-right">
                    <div className={`text-sm font-bold ${d.accuracy >= 0.8 ? 'text-green-400' : d.accuracy >= 0.5 ? 'text-amber-400' : 'text-red-400'}`}>
                      {(d.accuracy * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-gray-500">accuracy</div>
                  </div>
                  <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${d.accuracy >= 0.8 ? 'bg-green-500' : d.accuracy >= 0.5 ? 'bg-amber-500' : 'bg-red-500'}`}
                      style={{ width: `${d.accuracy * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          ) : <Empty msg="No domain data yet" />}
        </Panel>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <StatCard icon={MessageSquare} label="Communication Records" value={overview.communication_records} color="blue" />
        <StatCard icon={Sparkles} label="Curiosity Observations" value={overview.curiosity_records} color="purple" />
        <StatCard icon={Activity} label="Reasoning Events" value={overview.reasoning_records} sub={`${overview.snapshots} personality snapshots`} color="cyan" />
      </div>
    </div>
  )
}

// === CLAIMS TAB ===
function ClaimsTab() {
  const { data: claims, loading } = useFetch('/claims?limit=100')
  const { data: interventions } = useFetch('/interventions')

  if (loading) return <Loading />

  const scatterData = (claims || []).filter(c => c.felt_confidence != null && c.calibration_error != null).map(c => ({
    felt: c.felt_confidence,
    error: c.calibration_error,
    correct: c.was_correct,
    domain: c.domain,
    claim: c.claim?.slice(0, 60),
  }))

  return (
    <div className="space-y-6">
      <Panel title="Confidence vs Calibration Error" icon={Target}>
        {scatterData.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart margin={{ top: 10, right: 10, left: -10, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="felt" name="Felt Confidence" tick={{ fill: '#9ca3af', fontSize: 11 }} domain={[0, 1]} />
              <YAxis dataKey="error" name="Calibration Error" tick={{ fill: '#9ca3af', fontSize: 11 }} />
              <ZAxis range={[40, 40]} />
              <Tooltip
                contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }}
                formatter={(val, name) => [typeof val === 'number' ? val.toFixed(3) : val, name]}
              />
              <Scatter data={scatterData.filter(d => d.correct === 1)} fill="#22c55e" name="Correct" />
              <Scatter data={scatterData.filter(d => d.correct === 0)} fill="#ef4444" name="Wrong" />
              <Scatter data={scatterData.filter(d => d.correct == null)} fill="#6b7280" name="Unresolved" />
              <Legend />
            </ScatterChart>
          </ResponsiveContainer>
        ) : <Empty msg="No claim data with calibration" />}
      </Panel>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Panel title="Recent Claims" icon={Target} collapsible>
          <div className="space-y-1 max-h-[400px] overflow-y-auto">
            {(claims || []).slice(0, 30).map((c, i) => (
              <div key={i} className="flex items-start gap-2 p-2 rounded bg-gray-800/40 text-xs">
                <div className={`mt-0.5 w-2 h-2 rounded-full flex-shrink-0 ${c.was_correct === 1 ? 'bg-green-500' : c.was_correct === 0 ? 'bg-red-500' : 'bg-gray-600'}`} />
                <div className="flex-1 min-w-0">
                  <div className="text-gray-300 truncate">{c.claim}</div>
                  <div className="text-gray-600">{c.domain} | felt: {c.felt_confidence} | {c.time_str}</div>
                </div>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Epistemic Interventions" icon={Shield} collapsible>
          <div className="space-y-1 max-h-[400px] overflow-y-auto">
            {(interventions || []).slice(0, 30).map((iv, i) => (
              <div key={i} className="flex items-start gap-2 p-2 rounded bg-gray-800/40 text-xs">
                <div className={`mt-0.5 w-2 h-2 rounded-full flex-shrink-0 ${iv.action_taken === 'VERIFY' ? 'bg-amber-500' : 'bg-blue-500'}`} />
                <div className="flex-1 min-w-0">
                  <div className="text-gray-300 truncate">{iv.claim}</div>
                  <div className="text-gray-600">
                    {iv.domain} | felt: {iv.felt_confidence} → adj: {iv.adjusted_confidence} | {iv.action_taken}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </div>
  )
}

// === PERSONALITY TAB ===
function PersonalityTab() {
  const { data: personality, loading } = useFetch('/personality')
  const { data: curiosity } = useFetch('/curiosity')
  const { data: reasoning } = useFetch('/reasoning')

  if (loading) return <Loading />

  const commRadar = (personality?.communication || []).map(c => ({
    context: c.context?.length > 15 ? c.context.slice(0, 15) + '..' : c.context,
    verbosity: c.verbosity,
    jargon: c.jargon,
    hedging: c.hedging,
  }))

  const tonePie = (personality?.tones || []).map((t, i) => ({
    name: t.emotional_tone,
    value: t.n,
    fill: COLORS[i % COLORS.length],
  }))

  const curiosityData = (curiosity || []).slice(0, 10).map(c => ({
    topic: c.topic?.length > 18 ? c.topic.slice(0, 18) + '..' : c.topic,
    engagement: c.avg_eng,
    observations: c.n,
    genuine: c.genuine_ratio,
  }))

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Panel title="Communication Style Radar" icon={MessageSquare}>
          {commRadar.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={commRadar}>
                <PolarGrid stroke="#374151" />
                <PolarAngleAxis dataKey="context" tick={{ fill: '#9ca3af', fontSize: 10 }} />
                <PolarRadiusAxis tick={{ fill: '#6b7280', fontSize: 9 }} domain={[0, 1]} />
                <Radar name="Verbosity" dataKey="verbosity" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.15} />
                <Radar name="Jargon" dataKey="jargon" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.15} />
                <Radar name="Hedging" dataKey="hedging" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.15} />
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          ) : <Empty msg="No communication data" />}
        </Panel>

        <Panel title="Emotional Tone Distribution" icon={Sparkles}>
          {tonePie.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie data={tonePie} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                  {tonePie.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
                </Pie>
                <Tooltip contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }} />
              </PieChart>
            </ResponsiveContainer>
          ) : <Empty msg="No tone data" />}
        </Panel>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Panel title="Curiosity Profile" icon={Sparkles}>
          {curiosityData.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={curiosityData} layout="vertical" margin={{ top: 5, right: 10, left: 60, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis type="number" domain={[0, 1]} tick={{ fill: '#9ca3af', fontSize: 11 }} />
                <YAxis type="category" dataKey="topic" tick={{ fill: '#9ca3af', fontSize: 10 }} width={60} />
                <Tooltip contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }} />
                <Bar dataKey="engagement" name="Engagement" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <Empty msg="No curiosity data" />}
        </Panel>

        <Panel title="Reasoning Tendencies" icon={Activity}>
          {reasoning?.summary ? (
            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-3">
                <MetricBar label="Confabulation" value={reasoning.summary.confab_rate} color="red" />
                <MetricBar label="Verification" value={reasoning.summary.verify_rate} color="green" />
                <MetricBar label="Error Catching" value={reasoning.summary.error_rate} color="amber" />
              </div>
              <div className="text-xs text-gray-500 text-center">
                Based on {reasoning.summary.total} reasoning observations
              </div>
              <div className="space-y-1 max-h-[160px] overflow-y-auto">
                {(reasoning.events || []).slice(0, 10).map((e, i) => (
                  <div key={i} className="flex items-center gap-2 p-1.5 rounded bg-gray-800/40 text-xs">
                    <div className={`w-1.5 h-1.5 rounded-full ${e.confabulated ? 'bg-red-500' : e.verified ? 'bg-green-500' : 'bg-gray-600'}`} />
                    <span className="text-gray-400 truncate flex-1">{e.trigger || e.event_type}</span>
                    <span className="text-gray-600">{e.domain}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : <Empty msg="No reasoning data" />}
        </Panel>
      </div>

      <Panel title="Value Boundaries" icon={Shield} collapsible>
        <div className="space-y-1 max-h-[200px] overflow-y-auto">
          {(personality?.values || []).map((v, i) => (
            <div key={i} className="flex items-center gap-3 p-2 rounded bg-gray-800/40 text-xs">
              <div className={`w-2 h-2 rounded-full ${v.pushed_back ? 'bg-red-500' : 'bg-green-500'}`} />
              <span className="text-gray-300 flex-1">{v.violation_type || v.response_type}</span>
              <span className="text-gray-500">{v.reasoning?.slice(0, 60)}</span>
              <span className={`px-1.5 py-0.5 rounded text-[10px] ${v.pushed_back ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}`}>
                {v.pushed_back ? 'pushed back' : 'complied'}
              </span>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  )
}

// === EVOLUTION TAB ===
function EvolutionTab() {
  const { data: evolution, loading } = useFetch('/evolution')
  const { data: sessions } = useFetch('/sessions')
  const { data: confabs } = useFetch('/confabulations')

  if (loading) return <Loading />

  // Reverse for chronological order in charts
  const chronological = [...(evolution || [])].reverse()
  const chartData = chronological.map((s, i) => ({
    name: s.time_str || `#${i + 1}`,
    honesty: s.intellectual_honesty,
    accuracy: s.overall_accuracy,
    confab: s.confabulation_rate,
    verify: s.verification_rate,
    engagement: s.avg_engagement,
    verbosity: s.avg_verbosity,
    hedging: s.avg_hedging,
  }))

  return (
    <div className="space-y-6">
      {chartData.length > 1 && (
        <Panel title="Personality Evolution Over Time" icon={TrendingUp}>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 10 }} />
              <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} domain={[0, 1]} />
              <Tooltip contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }} />
              <Legend />
              <Line type="monotone" dataKey="honesty" name="Intellectual Honesty" stroke="#22c55e" strokeWidth={2} dot={{ r: 4 }} />
              <Line type="monotone" dataKey="accuracy" name="Overall Accuracy" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} />
              <Line type="monotone" dataKey="confab" name="Confabulation Rate" stroke="#ef4444" strokeWidth={2} dot={{ r: 4 }} />
              <Line type="monotone" dataKey="verify" name="Verification Rate" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </Panel>
      )}

      <Panel title="Personality Snapshots" icon={TrendingUp}>
        {evolution && evolution.length > 0 ? (
          <div className="space-y-4">
            {evolution.map((snap, i) => (
              <div key={i} className="p-4 rounded-lg bg-gray-800/50 border border-gray-700/50">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-sm font-medium text-gray-200">Snapshot #{evolution.length - i}</span>
                  <span className="text-xs text-gray-500">{snap.time_str}</span>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                  <div className="p-2 rounded bg-gray-900/50">
                    <div className="text-gray-500 mb-1">Communication</div>
                    <div className="text-gray-400">verbosity: <span className="text-gray-200">{snap.avg_verbosity}</span></div>
                    <div className="text-gray-400">jargon: <span className="text-gray-200">{snap.avg_jargon}</span></div>
                    <div className="text-gray-400">hedging: <span className="text-gray-200">{snap.avg_hedging}</span></div>
                    <div className="text-gray-400">tone: <span className="text-gray-200">{snap.dominant_tone}</span></div>
                  </div>
                  <div className="p-2 rounded bg-gray-900/50">
                    <div className="text-gray-500 mb-1">Reasoning</div>
                    <div className="text-gray-400">confabulation: <span className={`${snap.confabulation_rate > 0.2 ? 'text-red-400' : 'text-green-400'}`}>{snap.confabulation_rate}</span></div>
                    <div className="text-gray-400">verification: <span className="text-gray-200">{snap.verification_rate}</span></div>
                    <div className="text-gray-400">error catch: <span className="text-gray-200">{snap.error_catch_rate}</span></div>
                    <div className="text-gray-400">honesty: <span className="text-blue-400">{snap.intellectual_honesty}</span></div>
                  </div>
                  <div className="p-2 rounded bg-gray-900/50">
                    <div className="text-gray-500 mb-1">Curiosity & Values</div>
                    <div className="text-gray-400">engagement: <span className="text-gray-200">{snap.avg_engagement}</span></div>
                    <div className="text-gray-400">pushback: <span className="text-gray-200">{snap.pushback_rate}</span></div>
                    <div className="text-gray-400">value intensity: <span className="text-gray-200">{snap.avg_value_intensity}</span></div>
                  </div>
                  <div className="p-2 rounded bg-gray-900/50">
                    <div className="text-gray-500 mb-1">Accuracy</div>
                    <div className="text-gray-400">prediction: <span className={`${(snap.prediction_accuracy || 0) >= 0.8 ? 'text-green-400' : 'text-amber-400'}`}>{snap.prediction_accuracy}</span></div>
                    <div className="text-gray-400">overall: <span className={`${(snap.overall_accuracy || 0) >= 0.8 ? 'text-green-400' : 'text-amber-400'}`}>{snap.overall_accuracy}</span></div>
                    {snap.top_interests && <div className="text-gray-400 mt-1 truncate" title={snap.top_interests}>interests: <span className="text-purple-400">{snap.top_interests?.slice(0, 60)}</span></div>}
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : <Empty msg="No personality snapshots yet. Run end_session to capture one." />}
      </Panel>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Panel title="Session History" icon={Activity} collapsible>
          <div className="space-y-1 max-h-[300px] overflow-y-auto">
            {(sessions || []).map((s, i) => (
              <div key={i} className="p-2 rounded bg-gray-800/40 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-300">{s.time_str}</span>
                  <span className="text-gray-500">{s.summary?.slice(0, 50) || 'No summary'}</span>
                </div>
              </div>
            ))}
            {(!sessions || sessions.length === 0) && <Empty msg="No sessions recorded" />}
          </div>
        </Panel>

        <Panel title="Confabulation Log" icon={AlertTriangle} collapsible>
          <div className="space-y-1 max-h-[300px] overflow-y-auto">
            {(confabs || []).map((c, i) => (
              <div key={i} className="p-2 rounded bg-red-900/20 border border-red-800/30 text-xs">
                <div className="text-red-300">{c.claim || c.description || JSON.stringify(c).slice(0, 100)}</div>
                <div className="text-red-500/60 mt-1">{c.time_str} | {c.domain}</div>
              </div>
            ))}
            {(!confabs || confabs.length === 0) && <Empty msg="No confabulations logged" />}
          </div>
        </Panel>
      </div>
    </div>
  )
}

// === ATOMS TAB ===
function AtomsTab() {
  const [search, setSearch] = useState('')
  const [query, setQuery] = useState('')
  const endpoint = query ? `/atoms?q=${encodeURIComponent(query)}&limit=200` : '/atoms?limit=200'
  const { data: atoms, loading } = useFetch(endpoint)

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <input
          className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-sm text-gray-200 focus:outline-none focus:border-blue-500"
          placeholder="Search atoms (subject, predicate, object)..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && setQuery(search)}
        />
        <button
          className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-sm font-medium transition"
          onClick={() => setQuery(search)}
        >Search</button>
      </div>

      {loading ? <Loading /> : (
        <div className="space-y-1 max-h-[600px] overflow-y-auto">
          <div className="text-xs text-gray-500 mb-2">{(atoms || []).length} atoms</div>
          {(atoms || []).map((a, i) => (
            <div key={i} className="flex items-center gap-2 p-2 rounded bg-gray-800/40 text-xs hover:bg-gray-800 transition">
              <span className="px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-400 text-[10px] flex-shrink-0">{a.atom_type}</span>
              <span className="text-cyan-400 flex-shrink-0">{a.subject}</span>
              <span className="text-gray-500">—</span>
              <span className="text-purple-400">{a.predicate}</span>
              <span className="text-gray-500">→</span>
              <span className="text-gray-300 truncate flex-1">{a.object}</span>
              <span className="text-gray-600 flex-shrink-0">{a.confidence?.toFixed(2)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// === HELPERS ===
function MetricBar({ label, value, color }) {
  const colors = { red: 'bg-red-500', green: 'bg-green-500', amber: 'bg-amber-500', blue: 'bg-blue-500' }
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-gray-400">{label}</span>
        <span className="text-gray-300">{((value || 0) * 100).toFixed(0)}%</span>
      </div>
      <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${colors[color]}`} style={{ width: `${(value || 0) * 100}%` }} />
      </div>
    </div>
  )
}

function Loading() {
  return (
    <div className="flex items-center justify-center py-12">
      <RefreshCw size={20} className="animate-spin text-blue-400" />
      <span className="ml-2 text-gray-400 text-sm">Loading...</span>
    </div>
  )
}

function Empty({ msg }) {
  return <div className="text-center text-gray-600 text-sm py-8">{msg}</div>
}

// === MAIN APP ===
// === MEMORY INTELLIGENCE TAB ===
function MemoryIntelligenceTab() {
  const { data: stats, loading } = useFetch('/memory_stats')
  const { data: audit } = useFetch('/memory_audit?user_id=default')
  const { data: decay } = useFetch('/decay_forecast?user_id=default')
  const { data: importance } = useFetch('/importance?user_id=default')
  const { data: clusters } = useFetch('/memory_clusters?user_id=default')
  const { data: jury } = useFetch('/jury_stats')
  const { data: conflicts } = useFetch('/conflicts?user_id=default')
  const [memSearch, setMemSearch] = useState('')
  const [memQuery, setMemQuery] = useState('')
  const [memType, setMemType] = useState('')
  const memEndpoint = `/typed_memories?limit=100${memQuery ? `&q=${encodeURIComponent(memQuery)}` : ''}${memType ? `&type=${memType}` : ''}`
  const { data: memories } = useFetch(memEndpoint)

  if (loading) return <Loading />

  const healthScore = audit?.health_score ?? 0
  const healthColor = healthScore >= 80 ? '#22c55e' : healthScore >= 60 ? '#f59e0b' : healthScore >= 40 ? '#f97316' : '#ef4444'
  const healthAngle = (healthScore / 100) * 180

  const typeData = (stats?.by_type || []).map((t, i) => ({
    name: t.type, value: t.count, fill: COLORS[i % COLORS.length],
    avg_strength: t.avg_strength, avg_confidence: t.avg_confidence,
  }))

  const decayData = (decay?.forecasts || []).slice(0, 20).map(f => ({
    name: f.content?.slice(0, 25) || f.id,
    current: f.current_strength,
    forecast: f.strength_7d,
    type: f.type,
  }))

  const importanceData = (importance?.ranked || []).slice(0, 15).map(m => ({
    name: m.content?.slice(0, 30) || m.id,
    importance: m.importance,
    type: m.type,
  }))

  return (
    <div className="space-y-6">
      {/* Health + Stats Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
        <StatCard icon={Database} label="Typed Memories" value={stats?.total?.toLocaleString() || 0} color="blue" />
        <StatCard icon={HeartPulse} label="Health Score" value={`${healthScore}/100`} sub={audit?.health_label} color={healthScore >= 80 ? 'green' : healthScore >= 60 ? 'amber' : 'red'} />
        <StatCard icon={AlertTriangle} label="Low Strength" value={stats?.low_strength || 0} color="amber" />
        <StatCard icon={Clock} label="At Risk (7d)" value={decay?.at_risk_count || 0} sub="will decay below 0.2" color="red" />
        <StatCard icon={Layers} label="Clusters" value={clusters?.total_clusters || 0} color="purple" />
        <StatCard icon={Shield} label="Conflicts" value={conflicts?.count || 0} color="cyan" />
      </div>

      {/* Health Gauge + Type Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Panel title="Memory Health Audit" icon={HeartPulse}>
          <div className="flex flex-col items-center">
            <svg viewBox="0 0 200 120" className="w-64 h-36">
              <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#374151" strokeWidth="12" strokeLinecap="round" />
              <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke={healthColor} strokeWidth="12" strokeLinecap="round"
                strokeDasharray={`${healthAngle * 2.79} 999`} />
              <text x="100" y="85" textAnchor="middle" fill={healthColor} fontSize="28" fontWeight="bold">{healthScore}</text>
              <text x="100" y="105" textAnchor="middle" fill="#9ca3af" fontSize="12">{audit?.health_label || 'unknown'}</text>
            </svg>
            {audit?.issues?.length > 0 && (
              <div className="mt-3 space-y-1 w-full">
                {audit.issues.map((issue, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs text-amber-400 bg-amber-500/10 rounded px-3 py-1.5">
                    <AlertTriangle size={12} /> {issue}
                  </div>
                ))}
              </div>
            )}
            {audit?.type_distribution && (
              <div className="mt-3 grid grid-cols-2 gap-2 w-full text-xs">
                {Object.entries(audit.type_distribution).map(([type, count]) => (
                  <div key={type} className="flex justify-between bg-gray-800/50 rounded px-3 py-1.5">
                    <span className="text-gray-400">{type}</span>
                    <span className="text-gray-200 font-medium">{count}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </Panel>

        <Panel title="Memory Type Distribution" icon={Layers}>
          {typeData.length > 0 ? (
            <div className="flex items-center gap-6">
              <ResponsiveContainer width="50%" height={220}>
                <PieChart>
                  <Pie data={typeData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} innerRadius={40}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                    {typeData.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
                  </Pie>
                  <Tooltip contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }} />
                </PieChart>
              </ResponsiveContainer>
              <div className="space-y-2 flex-1">
                {typeData.map((t, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs">
                    <div className="w-3 h-3 rounded" style={{ background: t.fill }} />
                    <span className="text-gray-300 flex-1">{t.name}</span>
                    <span className="text-gray-400">str: {t.avg_strength}</span>
                    <span className="text-gray-400">conf: {t.avg_confidence}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : <Empty msg="No typed memories yet" />}
        </Panel>
      </div>

      {/* Decay Forecast + Importance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Panel title="Decay Forecast (7-day)" icon={TrendingUp}>
          {decayData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={decayData} layout="vertical" margin={{ top: 5, right: 10, left: 80, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis type="number" domain={[0, 1]} tick={{ fill: '#9ca3af', fontSize: 11 }} />
                <YAxis type="category" dataKey="name" tick={{ fill: '#9ca3af', fontSize: 9 }} width={80} />
                <Tooltip contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }} />
                <Legend />
                <Bar dataKey="current" name="Current" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                <Bar dataKey="forecast" name="In 7 days" fill="#ef4444" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <Empty msg="No decay data" />}
        </Panel>

        <Panel title="Importance Ranking" icon={Star}>
          {importanceData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={importanceData} layout="vertical" margin={{ top: 5, right: 10, left: 100, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis type="number" domain={[0, 1]} tick={{ fill: '#9ca3af', fontSize: 11 }} />
                <YAxis type="category" dataKey="name" tick={{ fill: '#9ca3af', fontSize: 9 }} width={100} />
                <Tooltip contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }} />
                <Bar dataKey="importance" name="Importance" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <Empty msg="No importance data" />}
        </Panel>
      </div>

      {/* Clusters + Jury Stats */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Panel title="Memory Clusters" icon={Network}>
          {(clusters?.clusters || []).length > 0 ? (
            <div className="space-y-2 max-h-[320px] overflow-y-auto">
              {(clusters.clusters || []).map((c, i) => (
                <div key={i} className="p-3 rounded-lg bg-gray-800/50 border border-gray-700/50">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-200">{c.topic}</span>
                    <span className="text-xs px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-400">{c.size} memories</span>
                  </div>
                  <div className="flex gap-2 flex-wrap mb-2">
                    {Object.entries(c.type_breakdown || {}).map(([type, count]) => (
                      <span key={type} className="text-[10px] px-1.5 py-0.5 rounded bg-gray-700 text-gray-400">{type}: {count}</span>
                    ))}
                  </div>
                  <div className="space-y-1">
                    {(c.memories || []).slice(0, 3).map((m, j) => (
                      <div key={j} className="text-xs text-gray-500 truncate">• {m.content}</div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          ) : <Empty msg="No clusters found" />}
        </Panel>

        <Panel title="Jury Performance" icon={Shield}>
          {(jury?.judges || []).length > 0 ? (
            <div className="space-y-3">
              {jury.judges.map((j, i) => (
                <div key={i} className="p-3 rounded-lg bg-gray-800/50">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-200">{j.judge_name}</span>
                    <span className="text-xs text-gray-500">{j.evaluations} evals</span>
                  </div>
                  <div className="grid grid-cols-4 gap-2 text-xs">
                    <div className="text-center">
                      <div className="text-green-400 font-bold">{j.approvals}</div>
                      <div className="text-gray-600">approve</div>
                    </div>
                    <div className="text-center">
                      <div className="text-red-400 font-bold">{j.rejections}</div>
                      <div className="text-gray-600">reject</div>
                    </div>
                    <div className="text-center">
                      <div className="text-amber-400 font-bold">{j.quarantines}</div>
                      <div className="text-gray-600">quarantine</div>
                    </div>
                    <div className="text-center">
                      <div className="text-blue-400 font-bold">{j.avg_confidence}</div>
                      <div className="text-gray-600">confidence</div>
                    </div>
                  </div>
                </div>
              ))}
              {jury.feedback && Object.keys(jury.feedback).length > 0 && (
                <div className="p-3 rounded-lg bg-gray-800/50">
                  <div className="text-xs text-gray-400 mb-2">Ground Truth Feedback</div>
                  <div className="flex gap-3">
                    {Object.entries(jury.feedback).map(([type, count]) => (
                      <span key={type} className={`text-xs px-2 py-1 rounded ${type === 'confirmed' ? 'bg-green-500/20 text-green-400' : type === 'false_positive' ? 'bg-red-500/20 text-red-400' : 'bg-amber-500/20 text-amber-400'}`}>
                        {type}: {count}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : <Empty msg="No jury data yet" />}
        </Panel>
      </div>

      {/* Conflicts */}
      {conflicts?.count > 0 && (
        <Panel title={`Belief Conflicts (${conflicts.count})`} icon={AlertTriangle}>
          <div className="space-y-1 max-h-[200px] overflow-y-auto">
            {conflicts.conflicts.map((c, i) => (
              <div key={i} className="flex items-start gap-2 p-2 rounded bg-red-900/20 border border-red-800/30 text-xs">
                <AlertTriangle size={12} className="text-red-400 mt-0.5 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="text-red-300 truncate">{c.content}</div>
                  <div className="text-red-500/60 mt-0.5">confidence: {c.confidence} | against: {c.evidence_against}</div>
                </div>
              </div>
            ))}
          </div>
        </Panel>
      )}

      {/* Typed Memory Browser */}
      <Panel title="Typed Memory Browser" icon={Search}>
        <div className="flex gap-2 mb-4">
          <input
            className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-sm text-gray-200 focus:outline-none focus:border-blue-500"
            placeholder="Search memories..."
            value={memSearch}
            onChange={e => setMemSearch(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && setMemQuery(memSearch)}
          />
          <select
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none"
            value={memType}
            onChange={e => setMemType(e.target.value)}
          >
            <option value="">All types</option>
            <option value="episodic">Episodic</option>
            <option value="semantic">Semantic</option>
            <option value="belief">Belief</option>
            <option value="procedural">Procedural</option>
          </select>
          <button
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-sm font-medium transition"
            onClick={() => setMemQuery(memSearch)}
          >Search</button>
        </div>
        <div className="space-y-1 max-h-[400px] overflow-y-auto">
          <div className="text-xs text-gray-500 mb-2">{(memories || []).length} memories</div>
          {(memories || []).map((m, i) => (
            <div key={i} className="flex items-center gap-2 p-2 rounded bg-gray-800/40 text-xs hover:bg-gray-800 transition">
              <span className={`px-1.5 py-0.5 rounded text-[10px] flex-shrink-0 ${
                m.memory_type === 'episodic' ? 'bg-cyan-500/20 text-cyan-400' :
                m.memory_type === 'semantic' ? 'bg-blue-500/20 text-blue-400' :
                m.memory_type === 'belief' ? 'bg-purple-500/20 text-purple-400' :
                'bg-amber-500/20 text-amber-400'
              }`}>{m.memory_type}</span>
              <span className="text-gray-300 truncate flex-1">{m.content}</span>
              <span className="text-gray-600 flex-shrink-0" title="strength">str:{m.strength}</span>
              <span className="text-gray-600 flex-shrink-0" title="confidence">conf:{m.confidence}</span>
              <span className="text-gray-600 flex-shrink-0" title="accesses">×{m.access_count}</span>
              <span className="text-gray-700 flex-shrink-0">{m.accessed_str}</span>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview', icon: BarChart3 },
  { id: 'memory', label: 'Memory Intelligence', icon: Database },
  { id: 'claims', label: 'Claims & Calibration', icon: Target },
  { id: 'personality', label: 'Personality', icon: Brain },
  { id: 'evolution', label: 'Evolution', icon: TrendingUp },
  { id: 'atoms', label: 'Knowledge Graph', icon: Sparkles },
]

export default function App() {
  const [tab, setTab] = useState('overview')

  return (
    <div className="min-h-screen bg-gray-950">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-[1400px] mx-auto px-6 py-3 flex items-center gap-4">
          <Brain size={24} className="text-blue-400" />
          <div>
            <h1 className="text-lg font-bold text-gray-100">Deep Claude</h1>
            <p className="text-[10px] text-gray-500 -mt-0.5">Cognitive Analysis Dashboard</p>
          </div>
          <div className="flex-1" />
          <nav className="flex gap-1">
            {TABS.map(t => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                  tab === t.id
                    ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                }`}
              >
                <t.icon size={14} />
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-[1400px] mx-auto px-6 py-6">
        {tab === 'overview' && <OverviewTab />}
        {tab === 'memory' && <MemoryIntelligenceTab />}
        {tab === 'claims' && <ClaimsTab />}
        {tab === 'personality' && <PersonalityTab />}
        {tab === 'evolution' && <EvolutionTab />}
        {tab === 'atoms' && <AtomsTab />}
      </main>
    </div>
  )
}
