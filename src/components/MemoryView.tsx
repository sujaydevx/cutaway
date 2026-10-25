import type { Arch } from '../lib/types';
import {
  GPUS,
  KV_PRECISIONS,
  PRECISIONS,
  decodeCeiling,
  flopsPerToken,
  gpusNeeded,
  kvBytesPerToken,
  memoryPlan,
} from '../lib/memory';
import { fmtBytes, fmtCtx } from '../lib/format';

export interface Settings {
  precision: string;
  kv: string;
  ctx: number;
  batch: number;
}

export const DEFAULT_SETTINGS: Settings = { precision: 'bf16', kv: 'kv16', ctx: 8192, batch: 1 };

const CTX_STEPS = Array.from({ length: 13 }, (_, i) => 512 * 2 ** i); // 512 .. 2M

export const bitsOf = (s: Settings) => (PRECISIONS.find((p) => p.id === s.precision) ?? PRECISIONS[1]).bits;
export const kvBitsOf = (s: Settings) => (KV_PRECISIONS.find((p) => p.id === s.kv) ?? KV_PRECISIONS[0]).bits;

export function Controls({ s, set, maxCtx }: { s: Settings; set: (s: Settings) => void; maxCtx: number | null }) {
  const stepIdx = Math.max(0, CTX_STEPS.findIndex((c) => c >= s.ctx));
  return (
    <div className="controls">
      <fieldset>
        <legend>Weights stored as</legend>
        <div className="seg">
          {PRECISIONS.map((p) => (
            <button key={p.id} className={s.precision === p.id ? 'on' : ''} onClick={() => set({ ...s, precision: p.id })}>
              {p.label}
            </button>
          ))}
        </div>
      </fieldset>
      <fieldset>
        <legend>KV cache stored as</legend>
        <div className="seg">
          {KV_PRECISIONS.map((p) => (
            <button key={p.id} className={s.kv === p.id ? 'on' : ''} onClick={() => set({ ...s, kv: p.id })}>
              {p.label}
            </button>
          ))}
        </div>
      </fieldset>
      <fieldset>
        <legend>
          Context length <span className="mono val">{fmtCtx(s.ctx)}</span>
          {maxCtx && s.ctx > maxCtx ? <span className="warn"> above this model’s {fmtCtx(maxCtx)} limit</span> : null}
        </legend>
        <input
          type="range"
          min={0}
          max={CTX_STEPS.length - 1}
          step={1}
          value={stepIdx}
          onChange={(e) => set({ ...s, ctx: CTX_STEPS[Number(e.target.value)] })}
          aria-label="Context length"
        />
      </fieldset>
      <fieldset>
        <legend>
          Concurrent sequences <span className="mono val">{s.batch}</span>
        </legend>
        <input type="range" min={0} max={7} step={1} value={Math.log2(s.batch)} onChange={(e) => set({ ...s, batch: 2 ** Number(e.target.value) })} aria-label="Batch size" />
      </fieldset>
    </div>
  );
}

export default function MemoryView({ arch, s, set }: { arch: Arch; s: Settings; set: (s: Settings) => void }) {
  const bits = bitsOf(s);
  const kvBits = kvBitsOf(s);
  const plan = memoryPlan(arch, bits, kvBits, s.ctx, s.batch);
  const kvTok = kvBytesPerToken(arch, kvBits);
  const parts = [
    { label: 'Weights', v: plan.weights, c: '#2B59C3' },
    { label: 'KV cache', v: plan.kv, c: '#D1651B' },
    { label: 'Runtime overhead', v: plan.overhead, c: '#9AA7B6' },
  ];
  return (
    <div className="memory">
      <Controls s={s} set={set} maxCtx={arch.ctx} />
      <div className="bar-block">
        <div className="bar-head">
          <span>Memory needed</span>
          <span className="mono">{fmtBytes(plan.total)}</span>
        </div>
        <div className="bar" style={{ height: 26 }}>
          {parts.map((p) => (
            <span key={p.label} title={`${p.label}: ${fmtBytes(p.v)}`} style={{ width: `${(p.v / plan.total) * 100}%`, background: p.c }} />
          ))}
        </div>
        <ul className="legend wide">
          {parts.map((p) => (
            <li key={p.label}>
              <span className="swatch" style={{ background: p.c }} />
              {p.label} <span className="mono">{fmtBytes(p.v)}</span>
            </li>
          ))}
        </ul>
        <p className="muted small">
          The KV cache costs {fmtBytes(kvTok)} per token ({arch.attn.kind === 'MLA' ? 'a compressed latent, shared by all heads' : `${arch.layers} layers × ${arch.attn.kvHeads} KV heads × ${arch.attn.headDim} dims × 2 (K and V)`}). About {(flopsPerToken(arch) / 1e9).toFixed(1)} GFLOPs of compute per generated token.
        </p>
      </div>

      <table className="table gpu">
        <thead>
          <tr>
            <th>GPU</th>
            <th className="num">Memory</th>
            <th>Fits</th>
            <th className="num">Decode ceiling</th>
          </tr>
        </thead>
        <tbody>
          {GPUS.map((g) => {
            const n = gpusNeeded(plan, g);
            const tps = decodeCeiling(arch, bits, g, n);
            return (
              <tr key={g.name} className={n === 1 ? 'fits' : ''}>
                <td>{g.name}</td>
                <td className="num mono">{g.vramGB} GB</td>
                <td>{n === 1 ? 'One card' : n <= 16 ? `${n} cards` : 'Not practical'}</td>
                <td className="num mono">{n <= 16 ? `${tps >= 100 ? Math.round(tps) : tps.toFixed(1)} tok/s` : '–'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="muted small">
        Decode ceiling assumes one sequence, every active weight read from memory once per token and no overhead from splitting across cards. Real throughput is lower. Treat it as an upper bound for comparing models, not a benchmark.
      </p>
    </div>
  );
}
