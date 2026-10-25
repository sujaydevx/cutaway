import type { Arch, Stage, TensorSpec } from '../lib/types';
import { GROUP_META, prod } from '../lib/arch';
import { fmtBytes, fmtParams, fmtPct, shapeStr } from '../lib/format';
import { ROLE_TEXT } from '../theme';
import type { Pick } from './ArchStack';

interface Props {
  arch: Arch;
  stage: Stage;
  pick: Pick;
}

function lookup(arch: Arch, stage: Stage, pick: NonNullable<Pick>): { tensors: TensorSpec[]; mult: number; prefix: string } {
  if (pick.scope === 'embed') return { tensors: arch.embed.filter((t) => t.key === pick.key), mult: 1, prefix: 'model.' };
  if (pick.scope === 'tail') {
    const t = arch.tail.filter((x) => x.key === pick.key);
    return { tensors: t, mult: 1, prefix: 'model.' };
  }
  const keys = pick.key.startsWith('moe.') && ['moe.gate', 'moe.up', 'moe.down'].includes(pick.key)
    ? ['moe.gate', 'moe.up', 'moe.down']
    : pick.key.startsWith('shared.') && ['shared.gate', 'shared.up', 'shared.down'].includes(pick.key)
      ? ['shared.gate', 'shared.up', 'shared.down']
      : [pick.key];
  return { tensors: stage.tensors.filter((t) => keys.includes(t.key)), mult: stage.count, prefix: `model.layers.${stage.from}${stage.count > 1 ? '–' + stage.to : ''}.` };
}

export default function Inspector({ arch, stage, pick }: Props) {
  if (!pick) {
    const rows = new Map<string, number>();
    stage.tensors.forEach((t) => rows.set(t.group, (rows.get(t.group) ?? 0) + t.count * prod(t.shape)));
    const list = Array.from(rows.entries()).sort((a, b) => b[1] - a[1]);
    return (
      <aside className="inspector" aria-live="polite">
        <h3>
          Layers {stage.from}
          {stage.count > 1 ? `–${stage.to}` : ''}
        </h3>
        <p className="muted">
          {stage.count} block{stage.count > 1 ? 's' : ''} with the same shape. One block holds {fmtParams(stage.paramsPerBlock)} parameters
          {arch.moe && stage.kind === 'moe' ? `, of which ${fmtParams(stage.activePerBlock)} run for each token` : ''}.
        </p>
        <ul className="split">
          {list.map(([g, v]) => (
            <li key={g}>
              <span className="swatch" style={{ background: GROUP_META[g as keyof typeof GROUP_META].color }} />
              <span className="split-name">{GROUP_META[g as keyof typeof GROUP_META].label}</span>
              <span className="mono">{fmtPct(v / stage.paramsPerBlock)}</span>
            </li>
          ))}
        </ul>
        <p className="hint">Click any box in either diagram to see its weight shape, size and what it does.</p>
      </aside>
    );
  }

  const { tensors, mult, prefix } = lookup(arch, stage, pick);
  if (!tensors.length) return null;
  const first = tensors[0];
  const color = GROUP_META[first.group].color;
  const groupParams = tensors.reduce((a, t) => a + t.count * prod(t.shape), 0) * mult;
  const isExpertGroup = tensors.length > 1;
  const title = isExpertGroup ? (first.group === 'experts' ? 'Routed experts' : 'Shared expert') : first.role === 'norm' ? arch.norm : first.name.replace(/\.weight$/, '');

  return (
    <aside className="inspector" aria-live="polite">
      <h3 style={{ borderLeftColor: color }}>{title}</h3>
      <p className="role">{ROLE_TEXT[first.role] ?? ''}</p>
      <dl className="facts">
        {tensors.map((t) => (
          <div key={t.key}>
            <dt className="mono">{t.name.replace(/\.weight$/, '')}</dt>
            <dd className="mono">
              {shapeStr(t.shape)}
              {t.count > 1 ? `  ×${t.count}` : ''}
            </dd>
          </div>
        ))}
      </dl>
      <dl className="facts">
        <div>
          <dt>Parameters{mult > 1 ? ` in all ${mult} blocks` : ''}</dt>
          <dd className="mono">{fmtParams(groupParams)}</dd>
        </div>
        <div>
          <dt>Share of whole model</dt>
          <dd className="mono">{fmtPct(groupParams / arch.totalParams, 2)}</dd>
        </div>
        <div>
          <dt>Size in BF16</dt>
          <dd className="mono">{fmtBytes(groupParams * 2)}</dd>
        </div>
        <div>
          <dt>Size at 4-bit</dt>
          <dd className="mono">{fmtBytes(groupParams * 0.5625)}</dd>
        </div>
        {first.activeCount !== first.count && (
          <div>
            <dt>Used per token</dt>
            <dd className="mono">
              {first.activeCount} of {first.count}
            </dd>
          </div>
        )}
      </dl>
      <p className="path mono">{prefix}{isExpertGroup ? first.name.split('.').slice(0, -2).join('.') + '.*' : first.name}</p>
    </aside>
  );
}
