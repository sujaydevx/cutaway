import type { LoadedModel } from '../lib/hf';
import type { Arch } from '../lib/types';
import { fmtBytes, fmtCtx, fmtParams, fmtPct } from '../lib/format';
import { kvBytesPerToken, memoryPlan } from '../lib/memory';
import { CompositionBar, Legend } from './ParamBars';
import ArchStack from './ArchStack';
import { Controls, bitsOf, kvBitsOf, type Settings } from './MemoryView';

interface Row {
  label: string;
  value: (a: Arch) => string;
  bar?: (a: Arch) => number;
}

export default function CompareView({
  models,
  s,
  set,
  onOpen,
  onRemove,
}: {
  models: LoadedModel[];
  s: Settings;
  set: (s: Settings) => void;
  onOpen: (key: string) => void;
  onRemove: (key: string) => void;
}) {
  const bits = bitsOf(s);
  const kvBits = kvBitsOf(s);
  const rows: Row[] = [
    { label: 'Family', value: (a) => a.modelType },
    { label: 'Total parameters', value: (a) => fmtParams(a.totalParams), bar: (a) => a.totalParams },
    { label: 'Used per token', value: (a) => (a.moe ? `${fmtParams(a.activeParams)} (${fmtPct(a.activeParams / a.totalParams, 0)})` : 'all of them'), bar: (a) => a.activeParams },
    { label: 'Layers', value: (a) => String(a.layers) },
    { label: 'Hidden size', value: (a) => String(a.hidden) },
    {
      label: 'Attention',
      value: (a) => (a.attn.kind === 'MLA' ? `MLA, ${a.attn.heads} heads` : `${a.attn.kind}, ${a.attn.heads} query / ${a.attn.kvHeads} KV heads`),
    },
    { label: 'Head size', value: (a) => String(a.attn.headDim) },
    {
      label: 'Feed-forward',
      value: (a) => (a.moe ? `${a.moe.experts} experts, top ${a.moe.topK}, ${a.moe.expertFfn} wide${a.moe.sharedFfn ? ' + shared' : ''}` : `${a.denseFfn} wide, ${a.gated ? 'gated' : 'plain'}`),
    },
    { label: 'Activation', value: (a) => a.act },
    { label: 'Normalisation', value: (a) => a.norm },
    { label: 'Positions', value: (a) => (a.ropeTheta ? `RoPE θ ${a.ropeTheta}${a.ropeScaling ? ', ' + a.ropeScaling : ''}` : 'learned') },
    { label: 'Context window', value: (a) => fmtCtx(a.ctx) },
    { label: 'Vocabulary', value: (a) => String(a.vocab) },
    { label: 'Tied embeddings', value: (a) => (a.tied ? 'yes' : 'no') },
    { label: 'KV cache per token', value: (a) => fmtBytes(kvBytesPerToken(a, kvBits)), bar: (a) => kvBytesPerToken(a, kvBits) },
    { label: `Weights at ${bits === 16 ? 'BF16' : bits === 32 ? 'FP32' : bits === 8 ? '8-bit' : '4-bit'}`, value: (a) => fmtBytes((a.totalParams * bits) / 8), bar: (a) => a.totalParams },
    { label: `Total at ${fmtCtx(s.ctx)} context`, value: (a) => fmtBytes(memoryPlan(a, bits, kvBits, s.ctx, s.batch).total), bar: (a) => memoryPlan(a, bits, kvBits, s.ctx, s.batch).total },
  ];

  const usable = models.filter((m) => m.arch);
  const maxCtx = usable.length ? Math.max(...usable.map((m) => m.arch!.ctx ?? 0)) : null;

  return (
    <div className="compare">
      <div className="compare-head">
        <h2>Comparing {models.length} models</h2>
        <p className="muted">Cells that differ from the first model are marked. Change the settings below to see how the memory rows move.</p>
      </div>
      <Controls s={s} set={set} maxCtx={maxCtx} />

      <div className="scroll">
        <table className="table compare-table">
          <thead>
            <tr>
              <th />
              {models.map((m) => (
                <th key={m.key}>
                  <button className="link-btn" onClick={() => onOpen(m.key)}>
                    {m.info.name}
                  </button>
                  <div className="muted small">{m.info.author}</div>
                  <button className="x" onClick={() => onRemove(m.key)} aria-label={`Remove ${m.info.name} from comparison`}>
                    Remove
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const vals = models.map((m) => (m.arch ? r.value(m.arch) : 'n/a'));
              const nums = r.bar ? models.map((m) => (m.arch ? r.bar!(m.arch) : 0)) : [];
              const mx = Math.max(...nums, 1);
              const allSame = vals.every((v) => v === vals[0]);
              return (
                <tr key={r.label}>
                  <th scope="row">{r.label}</th>
                  {vals.map((v, i) => (
                    <td key={models[i].key} className={!allSame && i > 0 && v !== vals[0] ? 'diff' : allSame ? 'same' : ''}>
                      <span className="mono">{v}</span>
                      {r.bar && models[i].arch && (
                        <span className="cellbar">
                          <span style={{ width: `${(nums[i] / mx) * 100}%` }} />
                        </span>
                      )}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <h3 className="section">Where the parameters live</h3>
      <div className="comp-list">
        {usable.map((m) => (
          <div key={m.key} className="comp-row">
            <div className="comp-name">
              {m.info.name}
              <span className="mono muted"> {fmtParams(m.arch!.totalParams)}</span>
            </div>
            <CompositionBar arch={m.arch!} which="params" height={18} />
          </div>
        ))}
        <Legend arch={mergeBreakdown(usable.map((m) => m.arch!))} />
      </div>

      <h3 className="section">Layer stacks, side by side</h3>
      <div className="scroll">
        <div className="stacks">
          {usable.map((m) => (
            <div key={m.key} className="stack-col">
              <div className="stack-title">{m.info.name}</div>
              <ArchStack name={m.info.name} arch={m.arch!} stageIdx={-1} pick={null} onStage={() => onOpen(m.key)} onPick={() => onOpen(m.key)} compact />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function mergeBreakdown(list: Arch[]): Arch {
  const seen = new Map<string, Arch['breakdown'][number]>();
  list.forEach((a) => a.breakdown.forEach((r) => seen.set(r.group, r)));
  return { ...list[0], breakdown: Array.from(seen.values()) } as Arch;
}
