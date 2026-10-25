import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { HubError, loadFromHub, loadPasted, loadSample, parseModelId, type LoadedModel } from './lib/hf';
import { SAMPLES } from './lib/samples';
import { fmtCount, fmtCtx, fmtParams, fmtPct, timeAgo } from './lib/format';
import { downloadPng, downloadSvg } from './lib/export';
import ArchStack, { type Pick } from './components/ArchStack';
import BlockDiagram from './components/BlockDiagram';
import Inspector from './components/Inspector';
import ParamBars, { Legend } from './components/ParamBars';
import MemoryView, { DEFAULT_SETTINGS, bitsOf, type Settings } from './components/MemoryView';
import CompareView from './components/CompareView';
import FlowPlayer from './components/FlowPlayer';
import { buildFlow } from './lib/flow';

type Tab = 'arch' | 'params' | 'memory' | 'tokenizer' | 'lineage' | 'files';
const TABS: { id: Tab; label: string }[] = [
  { id: 'arch', label: 'Architecture' },
  { id: 'params', label: 'Parameters' },
  { id: 'memory', label: 'Memory and hardware' },
  { id: 'tokenizer', label: 'Tokenizer' },
  { id: 'lineage', label: 'Lineage' },
  { id: 'files', label: 'Files and config' },
];

interface Problem {
  message: string;
  kind: HubError['kind'];
  id: string;
}

const hubUrl = (id: string) => `https://huggingface.co/${id}`;

function readHash(): { ids: string[]; view: 'model' | 'compare' } {
  const p = new URLSearchParams(window.location.hash.replace(/^#/, ''));
  const ids = (p.get('m') ?? '').split(',').filter(Boolean);
  return { ids, view: p.get('v') === 'compare' ? 'compare' : 'model' };
}

export default function App() {
  const [models, setModels] = useState<LoadedModel[]>([]);
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [view, setView] = useState<'model' | 'compare'>('model');
  const [tab, setTab] = useState<Tab>('arch');
  const [stageIdx, setStageIdx] = useState(0);
  const [pick, setPick] = useState<Pick>(null);
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState<{ id: string; step: string } | null>(null);
  const [problem, setProblem] = useState<Problem | null>(null);
  const [pasteOpen, setPasteOpen] = useState(false);
  const [pasteText, setPasteText] = useState('');
  const [pasteName, setPasteName] = useState('');
  const [fileFilter, setFileFilter] = useState('');
  const [copied, setCopied] = useState(false);

  const stackRef = useRef<SVGSVGElement>(null);
  const blockRef = useRef<SVGSVGElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const booted = useRef(false);

  const active = models.find((m) => m.key === activeKey) ?? null;
  const arch = active?.arch;

  // ---- data-flow walkthrough
  const [flow, setFlow] = useState({ on: false, i: 0, playing: false, speed: 1 });
  const reduced = useMemo(() => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false, []);
  const steps = useMemo(() => (arch ? buildFlow(arch) : []), [arch]);
  const step = flow.on ? steps[Math.min(flow.i, steps.length - 1)] : undefined;
  const flowGo = useCallback(
    (i: number) => setFlow((f) => ({ ...f, i: Math.max(0, Math.min(steps.length - 1, i)), playing: false })),
    [steps.length],
  );
  const flowToggle = useCallback(
    () =>
      setFlow((f) => {
        if (f.playing) return { ...f, playing: false };
        return { ...f, i: f.i >= steps.length - 1 ? 0 : f.i, playing: true };
      }),
    [steps.length],
  );
  const flowClose = useCallback(() => setFlow((f) => ({ ...f, on: false, playing: false, i: 0 })), []);

  const addModel = useCallback((m: LoadedModel, makeActive = true) => {
    setModels((cur) => (cur.some((x) => x.key === m.key) ? cur.map((x) => (x.key === m.key ? m : x)) : [...cur, m]));
    if (makeActive) {
      setActiveKey(m.key);
      setStageIdx(0);
      setPick(null);
    }
  }, []);

  const openId = useCallback(
    async (raw: string, makeActive = true) => {
      if (raw.startsWith('s:')) {
        const sid = raw.slice(2);
        if (SAMPLES.some((s) => s.id === sid)) addModel(loadSample(sid), makeActive);
        return;
      }
      const id = parseModelId(raw);
      if (!id) {
        setProblem({ kind: 'invalid', id: raw, message: 'That doesn’t look like a Hugging Face model link. Use something like huggingface.co/org/model-name.' });
        return;
      }
      setProblem(null);
      setBusy({ id, step: 'Contacting Hugging Face' });
      try {
        const m = await loadFromHub(id, (step) => setBusy({ id, step }));
        addModel(m, makeActive);
        setInput('');
      } catch (e) {
        const err = e instanceof HubError ? e : new HubError((e as Error).message, 'network');
        setProblem({ kind: err.kind, id, message: err.message });
        if (err.kind === 'gated' || err.kind === 'noconfig' || err.kind === 'network') {
          setPasteName(id);
        }
      } finally {
        setBusy(null);
      }
    },
    [addModel],
  );

  // a different model starts a fresh walkthrough; leaving the tab pauses it
  useEffect(() => {
    setFlow((f) => (f.on ? { ...f, on: false, playing: false, i: 0 } : f));
  }, [activeKey]);
  useEffect(() => {
    if (tab !== 'arch') setFlow((f) => (f.playing ? { ...f, playing: false } : f));
  }, [tab]);

  // autoplay: move to the next step after a reading pause
  useEffect(() => {
    if (!flow.on || !flow.playing || tab !== 'arch') return;
    const t = window.setTimeout(() => {
      setFlow((f) => (f.i >= steps.length - 1 ? { ...f, playing: false } : { ...f, i: f.i + 1 }));
    }, 3400 / flow.speed);
    return () => window.clearTimeout(t);
  }, [flow, steps.length, tab]);

  // show the right block, and keep the active box on screen
  useEffect(() => {
    if (!step) return;
    if (step.stageIdx !== undefined && step.stageIdx !== stageIdx) {
      setStageIdx(step.stageIdx);
      return;
    }
    const raf = requestAnimationFrame(() => {
      const el = document.querySelector(`.flow-${step.phase} .flow-ring`);
      if (!el) return;
      const r = el.getBoundingClientRect();
      const barBottom = document.querySelector('.flowbar')?.getBoundingClientRect().bottom ?? 0;
      const top = Math.max(barBottom, 0) + 24;
      const bottom = window.innerHeight - 24;
      if (r.top < top || r.bottom > bottom) {
        window.scrollBy({ top: r.top + r.height / 2 - (top + bottom) / 2, behavior: reduced ? 'auto' : 'smooth' });
      }
    });
    return () => cancelAnimationFrame(raf);
  }, [step, stageIdx, reduced]);

  useEffect(() => {
    if (!flow.on || tab !== 'arch') return;
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (/^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        setFlow((f) => ({ ...f, i: Math.min(steps.length - 1, f.i + 1), playing: false }));
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        setFlow((f) => ({ ...f, i: Math.max(0, f.i - 1), playing: false }));
      } else if (e.key === 'Escape') {
        flowClose();
      } else if (e.key === ' ' && el.tagName !== 'BUTTON') {
        e.preventDefault();
        flowToggle();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [flow.on, tab, steps.length, flowClose, flowToggle]);

  // first load: restore from the link, otherwise show a bundled sample
  useEffect(() => {
    if (booted.current) return;
    booted.current = true;
    const { ids, view: v } = readHash();
    if (ids.length === 0) {
      addModel(loadSample('meta-llama/Llama-3.1-8B'));
      return;
    }
    (async () => {
      for (let i = 0; i < ids.length; i++) await openId(ids[i], i === 0);
      if (v === 'compare' && ids.length > 1) setView('compare');
    })();
  }, [addModel, openId]);

  // keep the address bar in sync so a link reproduces the session
  useEffect(() => {
    if (!booted.current || models.length === 0) return;
    const ids = models.filter((m) => m.source !== 'pasted').map((m) => (m.source === 'sample' ? 's:' + m.id : m.id));
    const p = new URLSearchParams();
    p.set('m', ids.join(','));
    if (view === 'compare') p.set('v', 'compare');
    window.history.replaceState(null, '', '#' + p.toString().replace(/%2F/g, '/').replace(/%2C/g, ',').replace(/%3A/g, ':'));
  }, [models, view]);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) openId(input);
  };

  const submitPaste = () => {
    try {
      const m = loadPasted(pasteText, pasteName);
      addModel(m);
      setProblem(null);
      setPasteOpen(false);
      setPasteText('');
    } catch (e) {
      setProblem({ kind: 'invalid', id: pasteName, message: (e as Error).message });
    }
  };

  const remove = (key: string) => {
    setModels((cur) => {
      const next = cur.filter((m) => m.key !== key);
      if (key === activeKey) setActiveKey(next[0]?.key ?? null);
      if (next.length < 2) setView('model');
      return next;
    });
  };

  const copyLink = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard may be blocked; the address bar still has the link */
    }
  };

  const stage = arch ? arch.stages[Math.min(stageIdx, arch.stages.length - 1)] : null;
  const bits = bitsOf(settings);
  const fileName = (active?.info.name ?? 'model').replace(/[^\w.-]+/g, '_');

  const filteredTensors = useMemo(() => {
    if (!active?.tensors) return [];
    const q = fileFilter.trim().toLowerCase();
    const list = q ? active.tensors.filter((t) => t.name.toLowerCase().includes(q)) : active.tensors;
    return list;
  }, [active, fileFilter]);

  return (
    <div className="app">
      <header className="top">
        <div className="brand" aria-label="Cutaway">
          <svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true">
            <rect x="2" y="2" width="22" height="22" rx="2" fill="none" stroke="#142033" strokeWidth="1.6" />
            <path d="M2 17 L24 9" stroke="#142033" strokeWidth="1.6" />
            <rect x="2" y="17" width="22" height="7" fill="#142033" opacity="0.14" />
          </svg>
          <span>Cutaway</span>
        </div>
        <form className="urlbar" onSubmit={submit}>
          <label className="sr" htmlFor="url">
            Hugging Face model link
          </label>
          <input
            id="url"
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Paste a Hugging Face model link, e.g. huggingface.co/Qwen/Qwen3-30B-A3B"
            spellCheck={false}
            autoComplete="off"
          />
          <button type="submit" className="primary" disabled={!!busy || !input.trim()}>
            {busy ? 'Reading…' : 'Open model'}
          </button>
          <button type="button" onClick={() => setPasteOpen((v) => !v)} aria-expanded={pasteOpen}>
            Paste config.json
          </button>
        </form>
      </header>

      <div className="samples">
        <span className="muted">Samples that work offline</span>
        {SAMPLES.map((s) => (
          <button key={s.id} className="chip" onClick={() => openId('s:' + s.id)}>
            {s.short}
          </button>
        ))}
      </div>

      {busy && (
        <div className="status" role="status">
          <span className="spin" aria-hidden="true" /> {busy.step} for <b>{busy.id}</b>
        </div>
      )}
      {problem && (
        <div className="problem" role="alert">
          <p>{problem.message}</p>
          {(problem.kind === 'gated' || problem.kind === 'noconfig' || problem.kind === 'network') && !pasteOpen && (
            <button onClick={() => setPasteOpen(true)}>Paste config.json instead</button>
          )}
        </div>
      )}
      {pasteOpen && (
        <div className="paste">
          <label>
            Model name
            <input value={pasteName} onChange={(e) => setPasteName(e.target.value)} placeholder="org/model-name" />
          </label>
          <label>
            Contents of config.json
            <textarea value={pasteText} onChange={(e) => setPasteText(e.target.value)} rows={8} spellCheck={false} placeholder='{ "model_type": "llama", "hidden_size": 4096, … }' />
          </label>
          <div className="row">
            <button className="primary" onClick={submitPaste} disabled={!pasteText.trim()}>
              Draw it
            </button>
            <button onClick={() => setPasteOpen(false)}>Cancel</button>
          </div>
        </div>
      )}

      {models.length > 0 && (
        <nav className="models" aria-label="Loaded models">
          {models.map((m) => (
            <div key={m.key} className={'mtab' + (view === 'model' && m.key === activeKey ? ' on' : '')}>
              <button
                className="mtab-name"
                onClick={() => {
                  setActiveKey(m.key);
                  setView('model');
                  setStageIdx(0);
                  setPick(null);
                }}
              >
                {m.info.name}
                {m.arch && <span className="mono muted"> {fmtParams(m.arch.totalParams)}</span>}
              </button>
              <button className="mtab-x" onClick={() => remove(m.key)} aria-label={`Close ${m.info.name}`}>
                ×
              </button>
            </div>
          ))}
          {models.length > 1 && (
            <button className={'mtab compare-btn' + (view === 'compare' ? ' on' : '')} onClick={() => setView('compare')}>
              Compare all {models.length}
            </button>
          )}
          <span className="grow" />
          <button className="ghost" onClick={copyLink}>
            {copied ? 'Link copied' : 'Copy link to this view'}
          </button>
        </nav>
      )}

      <main>
        {view === 'compare' && models.length > 1 ? (
          <CompareView
            models={models}
            s={settings}
            set={setSettings}
            onOpen={(k) => {
              setActiveKey(k);
              setView('model');
              setStageIdx(0);
              setPick(null);
            }}
            onRemove={remove}
          />
        ) : active ? (
          <>
            {active.source === 'sample' && (
              <p className="banner">
                This is a bundled sample config, so it works without a network. Paste a link above to read a real model. Add a second model and a Compare tab appears.
              </p>
            )}
            <section className="head">
              <div className="title-row">
                <h1>{active.info.name}</h1>
                <a className="muted" href={hubUrl(active.info.id)} target="_blank" rel="noreferrer">
                  {active.info.author} on Hugging Face ↗
                </a>
              </div>
              <p className="meta">
                {[
                  active.info.license && `License ${active.info.license}`,
                  active.info.pipeline,
                  active.info.downloads !== undefined && `${fmtCount(active.info.downloads)} downloads`,
                  active.info.likes !== undefined && `${fmtCount(active.info.likes)} likes`,
                  active.info.lastModified && `updated ${timeAgo(active.info.lastModified)}`,
                ]
                  .filter(Boolean)
                  .join('   ·   ')}
              </p>

              {arch ? (
                <>
                  <dl className="specs">
                    <div>
                      <dt>Parameters</dt>
                      <dd>{fmtParams(arch.totalParams)}</dd>
                    </div>
                    {arch.moe && (
                      <div>
                        <dt>Used per token</dt>
                        <dd>{fmtParams(arch.activeParams)}</dd>
                      </div>
                    )}
                    <div>
                      <dt>Layers</dt>
                      <dd>{arch.layers}</dd>
                    </div>
                    <div>
                      <dt>Hidden size</dt>
                      <dd>{arch.hidden}</dd>
                    </div>
                    <div>
                      <dt>Attention</dt>
                      <dd>
                        {arch.attn.kind} {arch.attn.heads}
                        {arch.attn.kind === 'GQA' ? `/${arch.attn.kvHeads}` : ''}
                      </dd>
                    </div>
                    <div>
                      <dt>Context</dt>
                      <dd>{fmtCtx(arch.ctx)}</dd>
                    </div>
                    <div>
                      <dt>Vocabulary</dt>
                      <dd>{fmtCount(arch.vocab)}</dd>
                    </div>
                  </dl>
                  <p className="verify">{verifyLine(active)}</p>
                  {arch.notes.length > 0 && (
                    <ul className="notes">
                      {arch.notes.map((n) => (
                        <li key={n}>{n}</li>
                      ))}
                    </ul>
                  )}
                </>
              ) : (
                <div className="problem inline" role="alert">
                  <p>{active.archError}</p>
                  <p className="muted">The model page details and the raw config are still available below.</p>
                </div>
              )}
            </section>

            <div className="tabs" role="tablist">
              {TABS.map((t) => (
                <button key={t.id} role="tab" aria-selected={tab === t.id} className={tab === t.id ? 'on' : ''} onClick={() => setTab(t.id)}>
                  {t.label}
                </button>
              ))}
            </div>

            <section className="panel" role="tabpanel">
              {tab === 'arch' && arch && stage && (
                <>
                <FlowPlayer
                  steps={steps}
                  on={flow.on}
                  index={flow.i}
                  playing={flow.playing}
                  speed={flow.speed}
                  onOpen={() => setFlow((f) => ({ ...f, on: true, i: 0, playing: true }))}
                  onClose={flowClose}
                  onGo={flowGo}
                  onToggle={flowToggle}
                  onSpeed={(v) => setFlow((f) => ({ ...f, speed: v }))}
                />
                <div className="arch-grid">
                  <div className="col">
                    <div className="cap">
                      <h2>Whole model</h2>
                      <span className="tools">
                        <button className="ghost" onClick={() => downloadSvg(stackRef.current, fileName + '-stack')}>
                          SVG
                        </button>
                        <button className="ghost" onClick={() => downloadPng(stackRef.current, fileName + '-stack')}>
                          PNG
                        </button>
                      </span>
                    </div>
                    <ArchStack
                      name={active.info.name}
                      arch={arch}
                      stageIdx={stageIdx}
                      pick={flow.on ? null : pick}
                      onStage={setStageIdx}
                      onPick={setPick}
                      svgRef={stackRef}
                      active={step?.model}
                      motion={!!step && flow.playing && !reduced}
                      speed={flow.speed}
                      source={active.source === 'hub' ? 'config.json' : active.source}
                    />
                    <p className="legend-note">Click a block to open it. The thin bar at the bottom of each block shows where its weights sit.</p>
                    <Legend arch={arch} />
                  </div>
                  <div className="col">
                    <div className="cap">
                      <h2>Inside one block</h2>
                      <span className="tools">
                        <button className="ghost" onClick={() => downloadSvg(blockRef.current, fileName + '-block')}>
                          SVG
                        </button>
                        <button className="ghost" onClick={() => downloadPng(blockRef.current, fileName + '-block')}>
                          PNG
                        </button>
                      </span>
                    </div>
                    {arch.stages.length > 1 && (
                      <div className="seg stage-seg">
                        {arch.stages.map((s, i) => (
                          <button key={s.id} className={i === stageIdx ? 'on' : ''} onClick={() => { setStageIdx(i); setPick(null); }}>
                            {s.kind === 'moe' ? 'MoE' : 'Dense'} · layers {s.from}–{s.to}
                          </button>
                        ))}
                      </div>
                    )}
                    <BlockDiagram name={active.info.name} arch={arch} stage={stage} pick={flow.on ? null : pick} onPick={setPick} svgRef={blockRef} active={step?.block} motion={!!step && flow.playing && !reduced} speed={flow.speed} />
                  </div>
                  <div className="col side">
                    <div className="cap">
                      <h2>Inspector</h2>
                    </div>
                    <Inspector arch={arch} stage={stage} pick={pick} />
                  </div>
                </div>
                </>
              )}
              {tab === 'params' && arch && <ParamBars arch={arch} bits={bits} />}
              {tab === 'memory' && arch && <MemoryView arch={arch} s={settings} set={setSettings} />}
              {(tab === 'arch' || tab === 'params' || tab === 'memory') && !arch && <p className="muted">There is no drawing for this model because its config couldn’t be read as a transformer.</p>}
              {tab === 'tokenizer' && <TokenizerPanel m={active} />}
              {tab === 'lineage' && <LineagePanel m={active} onOpen={(id) => openId(id)} />}
              {tab === 'files' && (
                <div className="files">
                  <h3 className="section first">
                    Weights in {active.tensorFile ?? 'the first weight file'}
                  </h3>
                  {active.tensors ? (
                    <>
                      <input className="filter" value={fileFilter} onChange={(e) => setFileFilter(e.target.value)} placeholder={`Filter ${active.tensors.length} tensors, e.g. layers.0.`} />
                      <div className="scroll tall">
                        <table className="table">
                          <thead>
                            <tr>
                              <th>Tensor</th>
                              <th>Type</th>
                              <th className="num">Shape</th>
                            </tr>
                          </thead>
                          <tbody>
                            {filteredTensors.slice(0, 400).map((t) => (
                              <tr key={t.name}>
                                <td className="mono">{t.name}</td>
                                <td className="mono">{t.dtype}</td>
                                <td className="num mono">{t.shape.join(' × ')}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      {filteredTensors.length > 400 && <p className="muted small">Showing 400 of {filteredTensors.length}. Narrow the filter to see the rest.</p>}
                    </>
                  ) : (
                    <p className="muted">
                      {active.source === 'hub'
                        ? 'The weight headers could not be read from the browser, so shapes come from the config only.'
                        : 'Weight files are only read for models opened from a Hugging Face link.'}
                    </p>
                  )}
                  <h3 className="section">config.json</h3>
                  <pre className="code">{JSON.stringify(active.config, null, 2)}</pre>
                </div>
              )}
            </section>
          </>
        ) : (
          <p className="muted">Loading…</p>
        )}
      </main>
      <footer>
        <span>Shapes are derived from config.json and checked against the real weight files when the browser can read them.</span>
      </footer>
    </div>
  );
}

function verifyLine(m: LoadedModel): string {
  const a = m.arch;
  if (!a) return '';
  const parts: string[] = [`Derived ${fmtParams(a.totalParams)} parameters from the config`];
  const rep = m.info.reportedTotal;
  if (rep) {
    const delta = (a.totalParams - rep) / rep;
    parts.push(`Hugging Face reports ${fmtParams(rep)} (${Math.abs(delta) < 0.00005 ? 'exact match' : (delta > 0 ? '+' : '') + fmtPct(delta, 2) + ' difference'})`);
  }
  const v = m.verification;
  if (v && v.status !== 'unavailable') {
    parts.push(`${v.matched} of ${v.total} layer-0 weight shapes match ${v.file}`);
  } else if (v?.reason) {
    parts.push(v.reason);
  }
  return parts.join('. ') + '.';
}

function TokenizerPanel({ m }: { m: LoadedModel }) {
  const t = m.tokenizer;
  const rows: [string, string | number | undefined][] = [
    ['Vocabulary size', m.arch?.vocab ?? (m.config?.vocab_size as number | undefined)],
    ['Beginning of sequence', t?.bos],
    ['End of sequence', t?.eos],
    ['Padding', t?.pad],
    ['Unknown', t?.unk],
    ['Longest input it declares', t?.maxLength],
  ];
  if (!t && m.source !== 'hub') return <p className="muted">Tokenizer details are read from tokenizer_config.json, so they only appear for models opened from a Hugging Face link.</p>;
  return (
    <div className="tok">
      <dl className="facts wide">
        {rows
          .filter(([, v]) => v !== undefined)
          .map(([k, v]) => (
            <div key={k}>
              <dt>{k}</dt>
              <dd className="mono">{String(v)}</dd>
            </div>
          ))}
      </dl>
      {t?.chatTemplate ? (
        <>
          <h3 className="section">Chat template</h3>
          <p className="muted small">This Jinja template turns a list of chat messages into the exact text the model reads, special tokens included.</p>
          <pre className="code">{t.chatTemplate}</pre>
        </>
      ) : (
        <p className="muted">No chat template is published, so this is probably a base model for plain text completion.</p>
      )}
    </div>
  );
}

function LineagePanel({ m, onOpen }: { m: LoadedModel; onOpen: (id: string) => void }) {
  if (m.source !== 'hub') return <p className="muted">Lineage comes from the Hugging Face hub, so it only appears for models opened from a link.</p>;
  const list = (title: string, items: { id: string; note?: string }[], empty: string) => (
    <div className="lin-col">
      <h3>{title}</h3>
      {items.length === 0 ? (
        <p className="muted small">{empty}</p>
      ) : (
        <ul>
          {items.map((x) => (
            <li key={x.id}>
              <button className="link-btn" onClick={() => onOpen(x.id)}>
                {x.id}
              </button>
              {x.note && <span className="muted small mono"> {x.note}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
  return (
    <div className="lineage">
      {list(
        'Built from',
        m.info.baseModels.map((b) => ({ id: b.id, note: b.relation === 'base' ? '' : b.relation })),
        'No base model is listed. This is probably an original pre-trained model.',
      )}
      {list(
        'Most downloaded fine-tunes',
        (m.finetunes ?? []).map((f) => ({ id: f.id, note: f.downloads !== undefined ? fmtCount(f.downloads) + ' downloads' : '' })),
        'None found.',
      )}
      {list(
        'Most downloaded quantizations',
        (m.quantized ?? []).map((f) => ({ id: f.id, note: f.downloads !== undefined ? fmtCount(f.downloads) + ' downloads' : '' })),
        'None found.',
      )}
    </div>
  );
}
