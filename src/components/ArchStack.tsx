import type { ReactElement, Ref } from 'react';
import type { Arch, GroupKey, Stage } from '../lib/types';
import { GROUP_META } from '../lib/arch';
import { fmtParams, shapeStr } from '../lib/format';
import FlowMarks, { makeRegistry } from './FlowMarks';
import { FONT_MONO, FONT_UI, HAIR, INK, INK2, INK3, PLATE, SEL, tint } from '../theme';

export type Pick = { scope: 'embed' | 'tail' | 'stage'; key: string } | null;

interface Props {
  name: string;
  arch: Arch;
  stageIdx: number;
  pick: Pick;
  onStage: (i: number) => void;
  onPick: (p: Pick) => void;
  compact?: boolean;
  svgRef?: Ref<SVGSVGElement>;
  source?: string;
  /** anchor ids highlighted by the data-flow walkthrough */
  active?: string[];
  motion?: boolean;
  speed?: number;
}

function stageSummary(s: Stage, a: Arch): { title: string; sub: string } {
  const range = s.count === 1 ? `layer ${s.from}` : `layers ${s.from}–${s.to}`;
  const attn =
    a.attn.kind === 'MLA'
      ? `MLA ${a.attn.heads} heads`
      : `${a.attn.kind} ${a.attn.heads}${a.attn.kind === 'GQA' ? '/' + a.attn.kvHeads : ''} heads`;
  const mlp = s.kind === 'moe' && a.moe ? `MoE ${a.moe.experts} experts, top ${a.moe.topK}` : `MLP ${a.denseFfn}`;
  const title = a.moe ? (s.kind === 'moe' ? 'MoE block' : 'Dense block') : 'Decoder block';
  return { title: `${title}  (${range})`, sub: `${attn} · ${mlp}` };
}

export default function ArchStack({ name, arch, stageIdx, pick, onStage, onPick, compact, svgRef, source, active, motion = false, speed = 1 }: Props) {
  const { anchors, reg } = makeRegistry();
  const W = compact ? 230 : 420;
  const L = compact ? 46 : 54; // space for bracket
  const R = 12;
  const nodeX = L;
  const nodeW = W - L - R;
  const H_NODE = compact ? 34 : 40;
  const H_STAGE = compact ? 60 : 82;
  const H_ARROW = compact ? 22 : 30;
  const d = arch.hidden;

  type Item =
    | { t: 'text'; label: string; h: number }
    | { t: 'arrow'; shape: string; h: number }
    | { t: 'node'; label: string; sub?: string; group: GroupKey; scope: 'embed' | 'tail'; key: string; h: number; anchor?: string }
    | { t: 'stage'; i: number; h: number }
    | { t: 'title'; h: number };

  const items: Item[] = [];
  items.push({ t: 'text', label: 'next-token probabilities', h: 18 });
  items.push({ t: 'arrow', shape: `[B, T, ${arch.vocab}]`, h: H_ARROW });
  if (arch.tied) {
    items.push({ t: 'node', label: 'Output head', sub: 'shares weights with embeddings', group: 'head', scope: 'tail', key: 'final_norm', h: H_NODE, anchor: 'head' });
  } else {
    items.push({ t: 'node', label: 'Output head', sub: shapeStr([arch.vocab, d]), group: 'head', scope: 'tail', key: 'lm_head', h: H_NODE, anchor: 'head' });
  }
  items.push({ t: 'arrow', shape: `[B, T, ${d}]`, h: H_ARROW });
  items.push({ t: 'node', label: `Final ${arch.norm}`, sub: `[${d}]`, group: 'norm', scope: 'tail', key: 'final_norm', h: H_NODE, anchor: 'final_norm' });
  for (let i = arch.stages.length - 1; i >= 0; i--) {
    items.push({ t: 'arrow', shape: `[B, T, ${d}]`, h: H_ARROW });
    items.push({ t: 'stage', i, h: H_STAGE });
  }
  items.push({ t: 'arrow', shape: `[B, T, ${d}]`, h: H_ARROW });
  items.push({
    t: 'node',
    label: arch.embed.length > 1 ? 'Token + position embeddings' : 'Token embeddings',
    sub: shapeStr([arch.vocab, d]),
    group: 'embed',
    scope: 'embed',
    key: 'embed',
    h: H_NODE,
    anchor: 'embed',
  });
  items.push({ t: 'arrow', shape: '[B, T]', h: H_ARROW });
  items.push({ t: 'text', label: 'token ids', h: 18 });
  if (!compact) items.push({ t: 'title', h: 64 });

  const top = 10;
  const total = items.reduce((a, i) => a + i.h, 0) + top + 8;
  let y = top;
  const out: ReactElement[] = [];

  items.forEach((it, idx) => {
    const y0 = y;
    const cx = nodeX + nodeW / 2;
    if (it.t === 'text') reg(it.label.startsWith('next') ? 'probs' : 'ids', { x: nodeX, y: y0, w: nodeW, h: it.h });
    else if (it.t === 'node' && it.anchor) reg(it.anchor, { x: nodeX, y: y0, w: nodeW, h: it.h });
    else if (it.t === 'stage') reg(`stage:${it.i}`, { x: nodeX, y: y0, w: nodeW, h: it.h });
    if (it.t === 'text') {
      out.push(
        <text key={idx} x={cx} y={y0 + 13} textAnchor="middle" fontFamily={FONT_UI} fontSize={compact ? 11 : 12} fill={INK2} fontStyle="italic">
          {it.label}
        </text>,
      );
    } else if (it.t === 'arrow') {
      out.push(
        <g key={idx}>
          <line x1={cx} y1={y0 + it.h} x2={cx} y2={y0 + 5} stroke={INK2} strokeWidth={1.2} />
          <path d={`M ${cx - 4} ${y0 + 10} L ${cx} ${y0 + 3} L ${cx + 4} ${y0 + 10}`} fill="none" stroke={INK2} strokeWidth={1.2} />
          {!compact && (
            <text x={cx + 10} y={y0 + it.h / 2 + 4} fontFamily={FONT_MONO} fontSize={10.5} fill={INK3}>
              {it.shape}
            </text>
          )}
        </g>,
      );
    } else if (it.t === 'node') {
      const color = GROUP_META[it.group].color;
      const sel = pick && pick.scope === it.scope && pick.key === it.key;
      out.push(
        <g key={idx} style={{ cursor: 'pointer' }} onClick={() => onPick({ scope: it.scope, key: it.key })}>
          <rect x={nodeX} y={y0} width={nodeW} height={it.h} rx={3} fill={tint(color, 0.1)} stroke={sel ? SEL : color} strokeWidth={sel ? 2.4 : 1.2} />
          <text x={nodeX + 10} y={y0 + (compact ? 15 : 17)} fontFamily={FONT_UI} fontSize={compact ? 11.5 : 13} fontWeight={600} fill={INK}>
            {it.label}
          </text>
          {it.sub && (
            <text x={nodeX + 10} y={y0 + (compact ? 28 : 32)} fontFamily={FONT_MONO} fontSize={compact ? 9.5 : 10.5} fill={INK2}>
              {it.sub}
            </text>
          )}
        </g>,
      );
    } else if (it.t === 'stage') {
      const s = arch.stages[it.i];
      const sel = it.i === stageIdx;
      const { title, sub } = stageSummary(s, arch);
      const share = (g: GroupKey[]) =>
        s.tensors.filter((t) => g.includes(t.group)).reduce((a, t) => a + t.count * t.shape.reduce((p, q) => p * q, 1), 0) / s.paramsPerBlock;
      const segs: { g: GroupKey; f: number }[] = [
        { g: 'attn', f: share(['attn']) },
        { g: 'dense', f: share(['dense']) },
        { g: 'router', f: share(['router']) },
        { g: 'experts', f: share(['experts']) },
        { g: 'shared', f: share(['shared']) },
        { g: 'norm', f: share(['norm']) },
      ];
      const barW = nodeW - 20;
      let bx = nodeX + 10;
      const by = y0 + it.h - 14;
      out.push(
        <g key={idx} style={{ cursor: 'pointer' }} onClick={() => { onStage(it.i); onPick(null); }}>
          <rect x={nodeX} y={y0} width={nodeW} height={it.h} rx={3} fill={PLATE} stroke={sel ? SEL : INK} strokeWidth={sel ? 2.6 : 1.4} />
          <text x={nodeX + 10} y={y0 + (compact ? 16 : 20)} fontFamily={FONT_UI} fontSize={compact ? 11.5 : 13.5} fontWeight={650} fill={INK}>
            {compact ? title.split('  ')[0] : title}
          </text>
          {!compact && (
            <text x={nodeX + 10} y={y0 + 38} fontFamily={FONT_MONO} fontSize={10.5} fill={INK2}>
              {sub}
            </text>
          )}
          {!compact && (
            <text x={nodeX + 10} y={y0 + 54} fontFamily={FONT_MONO} fontSize={10.5} fill={INK3}>
              {fmtParams(s.paramsPerBlock)} params per block
            </text>
          )}
          {segs.map((sg) => {
            if (sg.f <= 0) return null;
            const w = sg.f * barW;
            const el = <rect key={sg.g} x={bx} y={by} width={Math.max(w - 1, 0.5)} height={6} fill={GROUP_META[sg.g].color} />;
            bx += w;
            return el;
          })}
          {/* bracket with repeat count */}
          <path d={`M ${nodeX - 8} ${y0 + 2} L ${nodeX - 13} ${y0 + 2} L ${nodeX - 13} ${y0 + it.h - 2} L ${nodeX - 8} ${y0 + it.h - 2}`} fill="none" stroke={INK} strokeWidth={1.2} />
          <text x={nodeX - 17} y={y0 + it.h / 2 + 4} textAnchor="end" fontFamily={FONT_MONO} fontSize={compact ? 11 : 13} fontWeight={500} fill={INK}>
            ×{s.count}
          </text>
        </g>,
      );
    } else if (it.t === 'title') {
      const tx = 10;
      const tw = W - 20;
      out.push(
        <g key={idx}>
          <rect x={tx} y={y0 + 4} width={tw} height={it.h - 8} fill="none" stroke={INK} strokeWidth={1} />
          <line x1={tx + tw * 0.62} y1={y0 + 4} x2={tx + tw * 0.62} y2={y0 + it.h - 4} stroke={INK} strokeWidth={1} />
          <text x={tx + 8} y={y0 + 22} fontFamily={FONT_UI} fontSize={13} fontWeight={650} fill={INK}>
            {name.length > 30 ? name.slice(0, 29) + '…' : name}
          </text>
          <text x={tx + 8} y={y0 + 39} fontFamily={FONT_MONO} fontSize={9.5} fill={INK2}>
            {arch.modelType} · {source ?? 'config.json'}
          </text>
          <text x={tx + tw * 0.62 + 8} y={y0 + 22} fontFamily={FONT_MONO} fontSize={11} fontWeight={500} fill={INK}>
            {fmtParams(arch.totalParams)}
          </text>
          <text x={tx + tw * 0.62 + 8} y={y0 + 39} fontFamily={FONT_MONO} fontSize={9.5} fill={INK2}>
            B = batch, T = tokens
          </text>
        </g>,
      );
    }
    y += it.h;
  });

  return (
    <svg
      ref={svgRef}
      xmlns="http://www.w3.org/2000/svg"
      viewBox={`0 0 ${W} ${total}`}
      width="100%"
      role="img"
      aria-label={`Layer stack of ${name}`}
      style={{ display: 'block', background: PLATE, maxWidth: W * (compact ? 1.2 : 1.15) }}
    >
      <rect x={0} y={0} width={W} height={total} fill={PLATE} stroke="none" />
      {out}
      {active && active.length > 0 && <FlowMarks anchors={anchors} ids={active} motion={motion} speed={speed} which="model" />}
      <rect x={0.5} y={0.5} width={W - 1} height={total - 1} fill="none" stroke={HAIR} />
    </svg>
  );
}
