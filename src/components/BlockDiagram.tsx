import type { ReactElement, Ref } from 'react';
import type { Arch, GroupKey, Stage, TensorSpec } from '../lib/types';
import { GROUP_META, prod } from '../lib/arch';
import { fmtParams, shapeStr } from '../lib/format';
import { FONT_MONO, FONT_UI, HAIR, INK, INK2, INK3, PLATE, SEL, tint } from '../theme';
import type { Pick } from './ArchStack';
import FlowMarks, { makeRegistry } from './FlowMarks';

interface Props {
  name: string;
  arch: Arch;
  stage: Stage;
  pick: Pick;
  onPick: (p: Pick) => void;
  svgRef?: Ref<SVGSVGElement>;
  /** anchor ids highlighted by the data-flow walkthrough */
  active?: string[];
  motion?: boolean;
  speed?: number;
}

const W = 660;
const RAIL = 34;
const CX = 365;

function hash(i: number) {
  // deterministic scatter so the "chosen" experts look spread out
  let x = (i + 1) * 2654435761;
  x ^= x >>> 15;
  return (x >>> 0) / 4294967296;
}

export default function BlockDiagram({ name, arch, stage, pick, onPick, svgRef, active, motion = false, speed = 1 }: Props) {
  const { anchors, reg } = makeRegistry();
  const d = arch.hidden;
  const a = arch.attn;
  const find = (k: string): TensorSpec | undefined => stage.tensors.find((t) => t.key === k);
  const sel = (k: string) => !!pick && pick.scope === 'stage' && pick.key === k;
  const el: ReactElement[] = [];
  let y = 8;
  let n = 0;
  const key = () => n++;

  const box = (opts: {
    k?: string;
    x?: number;
    w: number;
    h: number;
    title: string;
    sub?: string;
    group: GroupKey;
    params?: number;
    dashed?: boolean;
    a?: string;
  }) => {
    const x = opts.x ?? CX - opts.w / 2;
    if (opts.a) reg(opts.a, { x, y, w: opts.w, h: opts.h });
    const color = GROUP_META[opts.group].color;
    const s = opts.k ? sel(opts.k) : false;
    const k = opts.k;
    el.push(
      <g key={key()} style={{ cursor: k ? 'pointer' : 'default' }} onClick={k ? () => onPick({ scope: 'stage', key: k }) : undefined}>
        <rect
          x={x}
          y={y}
          width={opts.w}
          height={opts.h}
          rx={3}
          fill={tint(color, opts.dashed ? 0.04 : 0.1)}
          stroke={s ? SEL : color}
          strokeWidth={s ? 2.4 : 1.2}
          strokeDasharray={opts.dashed ? '4 3' : undefined}
        />
        <text x={x + 9} y={y + 16} fontFamily={FONT_UI} fontSize={12.5} fontWeight={600} fill={INK}>
          {opts.title}
        </text>
        {opts.sub && (
          <text x={x + 9} y={y + 31} fontFamily={FONT_MONO} fontSize={10.5} fill={INK2}>
            {opts.sub}
          </text>
        )}
        {opts.params !== undefined && (
          <text x={x + opts.w - 8} y={y + 16} textAnchor="end" fontFamily={FONT_MONO} fontSize={10.5} fill={INK3}>
            {fmtParams(opts.params)}
          </text>
        )}
      </g>,
    );
  };

  const arrowUp = (x: number, y1: number, y0: number) => {
    // from y1 (lower) up to y0 (upper)
    el.push(
      <g key={key()}>
        <line x1={x} y1={y1} x2={x} y2={y0 + 1} stroke={INK2} strokeWidth={1.2} />
        <path d={`M ${x - 3.5} ${y0 + 7} L ${x} ${y0} L ${x + 3.5} ${y0 + 7}`} fill="none" stroke={INK2} strokeWidth={1.2} />
      </g>,
    );
  };

  const arrowRow = (h = 20, label?: string) => {
    arrowUp(CX, y + h, y);
    if (label) {
      el.push(
        <text key={key()} x={CX + 10} y={y + h / 2 + 4} fontFamily={FONT_MONO} fontSize={10} fill={INK3}>
          {label}
        </text>,
      );
    }
    y += h;
  };

  /** lines from a single point below up to several boxes above (or the reverse merge) */
  const fan = (xs: number[], h = 22) => {
    const mid = y + h / 2;
    const lo = Math.min(...xs, CX);
    const hi = Math.max(...xs, CX);
    el.push(<line key={key()} x1={lo} y1={mid} x2={hi} y2={mid} stroke={INK2} strokeWidth={1.2} />);
    el.push(<line key={key()} x1={CX} y1={y + h} x2={CX} y2={mid} stroke={INK2} strokeWidth={1.2} />);
    xs.forEach((x) => arrowUp(x, mid, y));
    y += h;
  };

  /** several boxes below feed one element above */
  const fanIn = (xs: number[], h = 22) => {
    const mid = y + h / 2;
    const lo = Math.min(...xs, CX);
    const hi = Math.max(...xs, CX);
    el.push(<line key={key()} x1={lo} y1={mid} x2={hi} y2={mid} stroke={INK2} strokeWidth={1.2} />);
    xs.forEach((x) => el.push(<line key={key()} x1={x} y1={y + h} x2={x} y2={mid} stroke={INK2} strokeWidth={1.2} />));
    arrowUp(CX, mid, y);
    y += h;
  };

  /** rows are drawn top -> bottom, data flows bottom -> top */
  const railTop = y + 6;
  y += 20; // output label row
  reg('out', { x: CX - 112, y: y - 21, w: 224, h: 20 });
  el.push(
    <text key={key()} x={CX} y={y - 5} textAnchor="middle" fontFamily={FONT_MONO} fontSize={11} fill={INK2}>
      block output  [B, T, {d}]
    </text>,
  );

  const merge = (id: string, fromY?: number): number => {
    // output of a sublayer joins the residual stream
    const my = y + 16;
    el.push(<line key={key()} x1={CX} y1={y + 30} x2={CX} y2={my} stroke={INK2} strokeWidth={1.2} />);
    el.push(<line key={key()} x1={CX} y1={my} x2={RAIL + 9} y2={my} stroke={INK2} strokeWidth={1.2} />);
    el.push(
      <g key={key()}>
        <circle cx={RAIL} cy={my} r={9} fill={PLATE} stroke={INK} strokeWidth={1.3} />
        <line x1={RAIL - 4.5} y1={my} x2={RAIL + 4.5} y2={my} stroke={INK} strokeWidth={1.3} />
        <line x1={RAIL} y1={my - 4.5} x2={RAIL} y2={my + 4.5} stroke={INK} strokeWidth={1.3} />
      </g>,
    );
    reg(id, { x: RAIL - 9, y: my - 9, w: 18, h: 18, join: { fromY: fromY ?? my + 40, joinX: CX } });
    y += 30;
    return my;
  };

  const tap = (): number => {
    // residual stream is copied into the sublayer
    const ty = y + 13;
    el.push(<line key={key()} x1={RAIL} y1={ty} x2={CX} y2={ty} stroke={INK2} strokeWidth={1.2} />);
    el.push(<circle key={key()} cx={RAIL} cy={ty} r={2.5} fill={INK} />);
    arrowUp(CX, ty, y - 0);
    y += 26;
    return ty;
  };

  const normBox = (idx: number) => {
    const t = find(`norm.${idx}`);
    box({ k: `norm.${idx}`, a: `norm.${idx}`, w: 250, h: 34, title: arch.norm, sub: `[${d}]${t && arch.norm === 'LayerNorm' ? '  + bias' : ''}`, group: 'norm', params: t ? prod(t.shape) * (arch.norm === 'LayerNorm' ? 2 : 1) : undefined });
    y += 34;
  };

  // ======================= second half: feed-forward ====================
  const mergeBottom: number[] = [];
  mergeBottom.push(merge('add2'));

  if (stage.kind === 'moe' && arch.moe) {
    const m = arch.moe;
    const e = find('moe.up');
    const per = e ? 3 * prod(e.shape) : 0;
    box({ a: 'wsum', w: 380, h: 34, title: 'Weighted sum', sub: `top-${m.topK} expert outputs${m.sharedFfn ? ' + shared expert' : ''}`, group: 'experts' });
    y += 34;
    arrowRow(22);

    // expert panel: size it to the grid of squares so the footnote never overlaps them
    const hasShared = m.sharedFfn > 0;
    const panelW = hasShared ? 350 : 540;
    const panelX = hasShared ? 90 : CX - 270;
    const shown = Math.min(m.experts, 64);
    const cols = shown <= 16 ? shown : 16;
    const rows = Math.ceil(shown / cols);
    const cell = Math.min(18, (panelW - 20) / cols - 3);
    const panelH = Math.max(126, 44 + rows * (cell + 3) + 26);
    const gate = find('moe.gate');
    const s = sel('moe.gate') || sel('moe.up') || sel('moe.down');
    reg('experts', { x: panelX, y, w: panelW, h: panelH });
    el.push(
      <g key={key()} style={{ cursor: 'pointer' }} onClick={() => onPick({ scope: 'stage', key: 'moe.up' })}>
        <rect x={panelX} y={y} width={panelW} height={panelH} rx={3} fill={tint(GROUP_META.experts.color, 0.07)} stroke={s ? SEL : GROUP_META.experts.color} strokeWidth={s ? 2.4 : 1.2} />
        <text x={panelX + 9} y={y + 16} fontFamily={FONT_UI} fontSize={12.5} fontWeight={600} fill={INK}>
          {m.experts} routed experts, {m.topK} used per token
        </text>
        <text x={panelX + panelW - 8} y={y + 16} textAnchor="end" fontFamily={FONT_MONO} fontSize={10.5} fill={INK3}>
          {fmtParams(per * m.experts)}
        </text>
        <text x={panelX + 9} y={y + 31} fontFamily={FONT_MONO} fontSize={10.5} fill={INK2}>
          each: gate/up {shapeStr(gate?.shape ?? [])}, down {shapeStr(find('moe.down')?.shape ?? [])}
        </text>
      </g>,
    );
    const per_sq = m.experts / shown;
    const chosen = new Set<number>();
    const want = Math.max(1, Math.round((m.topK / m.experts) * shown));
    const order = Array.from({ length: shown }, (_, i) => i).sort((p, q) => hash(p) - hash(q));
    order.slice(0, want).forEach((i) => chosen.add(i));
    const gridW = cols * (cell + 3) - 3;
    const gx = panelX + (panelW - gridW) / 2;
    const gy = y + 44;
    for (let i = 0; i < shown; i++) {
      const cx = gx + (i % cols) * (cell + 3);
      const cy = gy + Math.floor(i / cols) * (cell + 3);
      const on = chosen.has(i);
      el.push(<rect key={key()} x={cx} y={cy} width={cell} height={cell} rx={2} fill={on ? GROUP_META.experts.color : PLATE} stroke={GROUP_META.experts.color} strokeWidth={1} opacity={on ? 1 : 0.75} />);
    }
    const noteY = gy + rows * (cell + 3) + 13;
    el.push(
      <text key={key()} x={panelX + 9} y={noteY} fontFamily={FONT_UI} fontSize={10.5} fill={INK2}>
        {per_sq > 1 ? `Each square = ${Math.round(per_sq)} experts. ` : ''}Filled = chosen for one token (example).
      </text>,
    );
    if (hasShared) {
      const sx = panelX + panelW + 20;
      const sw = W - sx - 12;
      const st = find('shared.up');
      box({
        k: 'shared.up',
        a: 'shared',
        x: sx,
        w: sw,
        h: panelH,
        title: 'Shared expert',
        sub: `runs for every token`,
        group: 'shared',
        params: st ? 3 * prod(st.shape) : undefined,
      });
      el.push(
        <text key={key()} x={sx + 9} y={y + 50} fontFamily={FONT_MONO} fontSize={10.5} fill={INK2}>
          gate/up {shapeStr(st?.shape ?? [])}
        </text>,
        <text key={key()} x={sx + 9} y={y + 65} fontFamily={FONT_MONO} fontSize={10.5} fill={INK2}>
          down {shapeStr(find('shared.down')?.shape ?? [])}
        </text>,
      );
    }
    y += panelH;
    arrowRow(22);
    const r = find('moe.router');
    box({ k: 'moe.router', a: 'router', w: 330, h: 44, title: 'Router', sub: `${shapeStr(r?.shape ?? [])}  → scores for ${m.experts} experts`, group: 'router', params: r ? prod(r.shape) : undefined });
    y += 44;
  } else {
    const f = arch.denseFfn;
    const down = find('mlp.down');
    box({ k: 'mlp.down', a: 'mlp.down', w: 320, h: 44, title: arch.gated ? 'down_proj' : 'fc_out', sub: `${shapeStr(down?.shape ?? [])}  (${f} → ${d})`, group: 'dense', params: down ? prod(down.shape) : undefined });
    y += 44;
    arrowRow(20, `[B, T, ${f}]`);
    if (arch.gated) {
      box({ a: 'mlp.act', w: 380, h: 34, title: `${arch.act}(gate) ⊙ up`, sub: 'element-wise product', group: 'dense', dashed: true });
      y += 34;
      const g = find('mlp.gate');
      const u = find('mlp.up');
      fanIn([CX - 130, CX + 130], 22);
      box({ k: 'mlp.gate', a: 'mlp.expand', x: CX - 130 - 115, w: 230, h: 48, title: 'gate_proj', sub: `${shapeStr(g?.shape ?? [])}`, group: 'dense', params: g ? prod(g.shape) : undefined });
      box({ k: 'mlp.up', a: 'mlp.expand', x: CX + 130 - 115, w: 230, h: 48, title: 'up_proj', sub: `${shapeStr(u?.shape ?? [])}`, group: 'dense', params: u ? prod(u.shape) : undefined });
      y += 48;
    } else {
      box({ a: 'mlp.act', w: 260, h: 30, title: arch.act, sub: undefined, group: 'dense', dashed: true });
      y += 30;
      arrowRow(18);
      const u = find('mlp.up');
      box({ k: 'mlp.up', a: 'mlp.expand', w: 320, h: 44, title: 'fc_in', sub: `${shapeStr(u?.shape ?? [])}  (${d} → ${f})`, group: 'dense', params: u ? prod(u.shape) : undefined });
      y += 44;
    }
  }
  arrowRow(22);
  normBox(1);
  const tapY2 = tap();

  // ======================= first half: attention ========================
  mergeBottom.push(merge('add1'));
  const o = find('attn.o');
  box({ k: 'attn.o', a: 'attn.o', w: 320, h: 44, title: 'o_proj', sub: `${shapeStr(o?.shape ?? [])}  (heads → ${d})`, group: 'attn', params: o ? prod(o.shape) : undefined });
  y += 44;
  arrowRow(20, `[B, T, ${a.heads * (a.mla?.vHead ?? a.headDim)}]`);

  if (a.kind === 'MLA' && a.mla) {
    const m = a.mla;
    box({
      w: 440,
      h: 62,
      title: 'Scaled dot-product attention',
      a: 'sdpa',
      sub: `${a.heads} heads · q/k dim ${m.qkNope}+${m.qkRope} (content + rotary) · v dim ${m.vHead}`,
      group: 'attn',
      dashed: true,
    });
    el.push(
      <text key={key()} x={CX - 220 + 9} y={y + 47} fontFamily={FONT_UI} fontSize={10.5} fill={INK2}>
        Every head attends to the same compressed cache of {m.kvLora + m.qkRope} numbers per token.
      </text>,
    );
    y += 62;
    fanIn([CX - 150, CX + 150], 22);
    // two columns, listed top -> bottom
    const colW = 270;
    const lx = CX - 150 - colW / 2;
    const rx = CX + 150 - colW / 2;
    const colTop = y;
    const draw = (x: number, items: { k: string; title: string; a: string }[]) => {
      let cy = colTop;
      items.forEach((it, i) => {
        const t = find(it.k);
        const saveY = y;
        y = cy;
        box({ k: it.k, a: it.a, x, w: colW, h: 40, title: it.title, sub: shapeStr(t?.shape ?? []), group: 'attn', params: t ? prod(t.shape) : undefined });
        y = saveY;
        cy += 40;
        if (i < items.length - 1) {
          arrowUp(x + colW / 2, cy + 14, cy);
          cy += 14;
        }
      });
      return cy;
    };
    const qItems = find('attn.q_b')
      ? [
          { k: 'attn.q_b', a: 'mla.up', title: 'q_b_proj  (expand)' },
          { k: 'attn.q_a_norm', a: 'mla.norm', title: 'q_a_layernorm' },
          { k: 'attn.q_a', a: 'mla.down', title: 'q_a_proj  (compress)' },
        ]
      : [{ k: 'attn.q', a: 'mla.down', title: 'q_proj' }];
    const kvItems = [
      { k: 'attn.kv_b', a: 'mla.up', title: 'kv_b_proj  (expand)' },
      { k: 'attn.kv_a_norm', a: 'mla.norm', title: 'kv_a_layernorm' },
      { k: 'attn.kv_a', a: 'mla.down', title: 'kv_a_proj  (compress, cached)' },
    ];
    const e1 = draw(lx, qItems);
    const e2 = draw(rx, kvItems);
    y = Math.max(e1, e2);
    fan([CX - 150, CX + 150], 22);
  } else {
    const hs = a.heads;
    const kvh = a.kvHeads;
    box({
      w: 460,
      h: 70,
      title: 'Scaled dot-product attention',
      a: 'sdpa',
      sub: `softmax( Q·Kᵀ / √${a.headDim} ) · V   ·  causal mask`,
      group: 'attn',
      dashed: true,
    });
    el.push(
      <text key={key()} x={CX - 230 + 9} y={y + 47} fontFamily={FONT_MONO} fontSize={10.5} fill={INK2}>
        Q [B, {hs}, T, {a.headDim}]   K, V [B, {kvh}, T, {a.headDim}]
      </text>,
      <text key={key()} x={CX - 230 + 9} y={y + 62} fontFamily={FONT_UI} fontSize={10.5} fill={INK2}>
        {a.kind === 'MHA' ? 'Every head has its own keys and values.' : a.kind === 'MQA' ? 'All query heads share one key/value head.' : `${hs / kvh} query heads share each of the ${kvh} key/value heads.`}
      </text>,
    );
    y += 70;
    if (arch.ropeTheta) {
      arrowRow(16);
      const rs = arch.ropeScaling ? `, ${arch.ropeScaling} scaling` : '';
      box({ a: 'rope', w: 360, h: 26, title: `RoPE  θ = ${arch.ropeTheta}${rs}`, group: 'norm', dashed: true });
      y += 26;
    }
    if (a.qkNorm) {
      arrowRow(16);
      box({ k: 'attn.q_norm', a: 'qknorm', x: CX - 100 - 85, w: 170, h: 32, title: 'q_norm', sub: `[${a.headDim}]`, group: 'norm' });
      box({ k: 'attn.k_norm', a: 'qknorm', x: CX + 100 - 85, w: 170, h: 32, title: 'k_norm', sub: `[${a.headDim}]`, group: 'norm' });
      y += 32;
    }
    const xs = [CX - 200, CX, CX + 200];
    const q = find('attn.q');
    const k = find('attn.k');
    const v = find('attn.v');
    fanIn(xs, 22);
    const qkvW = 184;
    const qkv: [string, string, TensorSpec | undefined, number][] = [
      ['attn.q', 'q_proj', q, xs[0]],
      ['attn.k', 'k_proj', k, xs[1]],
      ['attn.v', 'v_proj', v, xs[2]],
    ];
    qkv.forEach(([kk, title, t, cx]) => {
      box({ k: kk, a: 'qkv', x: cx - qkvW / 2, w: qkvW, h: 52, title: title + (a.qkvBias ? ' + bias' : ''), sub: shapeStr(t?.shape ?? []), group: 'attn', params: t ? prod(t.shape) : undefined });
      el.push(
        <text key={key()} x={cx - qkvW / 2 + 9} y={y + 45} fontFamily={FONT_UI} fontSize={10} fill={INK3}>
          {kk === 'attn.q' ? `${hs} heads × ${a.headDim}` : `${kvh} heads × ${a.headDim}`}
        </text>,
      );
    });
    y += 52;
    // the normalised stream feeds q, k and v
    fan(xs, 22);
  }

  normBox(0);
  const tapY1 = tap();
  reg('in', { x: CX - 112, y: y - 1, w: 224, h: 20 });
  el.push(
    <text key={key()} x={CX} y={y + 11} textAnchor="middle" fontFamily={FONT_MONO} fontSize={11} fill={INK2}>
      block input  [B, T, {d}]
    </text>,
  );
  y += 22;
  const railBottom = y - 8;

  // residual rail, drawn first so it sits behind the other marks
  const rail = (
    <g key="rail">
      <line x1={RAIL} y1={railBottom} x2={RAIL} y2={railTop} stroke={INK} strokeWidth={1.6} />
      <path d={`M ${RAIL - 4.5} ${railTop + 8} L ${RAIL} ${railTop} L ${RAIL + 4.5} ${railTop + 8}`} fill="none" stroke={INK} strokeWidth={1.6} />
      <text transform={`translate(${RAIL - 14} ${(railTop + railBottom) / 2}) rotate(-90)`} textAnchor="middle" fontFamily={FONT_UI} fontSize={11} fill={INK2}>
        residual stream
      </text>
    </g>
  );

  const H = y + 8;
  void mergeBottom;
  // the rail origin for each "add" is the tap that fed its sub-layer
  const fixJoin = (id: string, fromY: number) => anchors[id]?.forEach((r) => r.join && (r.join.fromY = fromY));
  fixJoin('add2', tapY2);
  fixJoin('add1', tapY1);

  return (
    <svg
      ref={svgRef}
      xmlns="http://www.w3.org/2000/svg"
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      role="img"
      aria-label={`Inside one block of ${name}`}
      style={{ display: 'block', background: PLATE, maxWidth: W * 1.2 }}
    >
      <rect x={0} y={0} width={W} height={H} fill={PLATE} />
      {rail}
      {el}
      {active && active.length > 0 && <FlowMarks anchors={anchors} ids={active} motion={motion} speed={speed} which="block" />}
      <rect x={0.5} y={0.5} width={W - 1} height={H - 1} fill="none" stroke={HAIR} />
    </svg>
  );
}
