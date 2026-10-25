import type { FlowStep } from '../lib/flow';

interface Props {
  steps: FlowStep[];
  on: boolean;
  index: number;
  playing: boolean;
  speed: number;
  onOpen: () => void;
  onClose: () => void;
  onGo: (i: number) => void;
  onToggle: () => void;
  onSpeed: (s: number) => void;
}

const Icon = {
  play: (
    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
      <path d="M2.5 1.5 L10 6 L2.5 10.5 Z" fill="currentColor" />
    </svg>
  ),
  pause: (
    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
      <rect x="2" y="1.5" width="3" height="9" fill="currentColor" />
      <rect x="7" y="1.5" width="3" height="9" fill="currentColor" />
    </svg>
  ),
  prev: (
    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
      <rect x="1.5" y="2" width="1.8" height="8" fill="currentColor" />
      <path d="M10.5 1.8 L4.6 6 L10.5 10.2 Z" fill="currentColor" />
    </svg>
  ),
  next: (
    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
      <rect x="8.7" y="2" width="1.8" height="8" fill="currentColor" />
      <path d="M1.5 1.8 L7.4 6 L1.5 10.2 Z" fill="currentColor" />
    </svg>
  ),
};

export default function FlowPlayer({ steps, on, index, playing, speed, onOpen, onClose, onGo, onToggle, onSpeed }: Props) {
  if (!on) {
    return (
      <div className="flow-idle">
        <button className="primary flow-start" onClick={onOpen}>
          {Icon.play}
          <span>Follow a token through the model</span>
        </button>
        <span className="muted small">Steps through every layer and shows the tensor shape at each point. Arrow keys step, space plays.</span>
      </div>
    );
  }
  const i = Math.min(index, steps.length - 1);
  const s = steps[i];
  const last = i === steps.length - 1;
  return (
    <div className="flowbar" role="region" aria-label="Data flow walkthrough">
      <div className="flow-ctl">
        <button onClick={() => onGo(i - 1)} disabled={i === 0} aria-label="Previous step" className="icon">
          {Icon.prev}
        </button>
        <button className="primary icon-text" onClick={onToggle} aria-label={playing ? 'Pause' : last ? 'Replay' : 'Play'}>
          {playing ? Icon.pause : Icon.play}
          <span>{playing ? 'Pause' : last ? 'Replay' : 'Play'}</span>
        </button>
        <button onClick={() => onGo(i + 1)} disabled={last} aria-label="Next step" className="icon">
          {Icon.next}
        </button>
        <div className="seg" role="group" aria-label="Speed">
          {[0.5, 1, 2].map((v) => (
            <button key={v} className={v === speed ? 'on' : ''} onClick={() => onSpeed(v)} aria-pressed={v === speed}>
              {v}×
            </button>
          ))}
        </div>
        <span className="grow" />
        <span className="flow-count">
          Step {i + 1} of {steps.length}
          <span className="muted"> · {s.phase === 'model' ? 'whole model' : 'inside the block'}</span>
        </span>
        <button className="ghost" onClick={onClose}>
          Close
        </button>
      </div>
      <div className="flow-track" role="group" aria-label="Steps">
        {steps.map((st, k) => (
          <button
            key={st.id}
            className={`tick ${st.phase}${k === i ? ' now' : k < i ? ' done' : ''}`}
            onClick={() => onGo(k)}
            title={`${k + 1}. ${st.title}`}
            aria-label={`Step ${k + 1}: ${st.title}`}
            aria-current={k === i ? 'step' : undefined}
          />
        ))}
      </div>
      <div className="flow-cap" aria-live="polite">
        <div className="flow-text">
          <h3>{s.title}</h3>
          <p>{s.body}</p>
        </div>
        <dl className="flow-shapes">
          {s.from && (
            <div>
              <dt>In</dt>
              <dd>{s.from}</dd>
            </div>
          )}
          {s.to && (
            <div>
              <dt>{s.from ? 'Out' : 'Shape'}</dt>
              <dd>{s.to}</dd>
            </div>
          )}
          <div className="flow-key">B = batch size, T = number of tokens</div>
        </dl>
      </div>
    </div>
  );
}
