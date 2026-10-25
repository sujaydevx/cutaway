import { INK, SEL } from '../theme';

/** A box the diagrams register while laying themselves out, so the walkthrough can point at it. */
export interface Anchor {
  x: number;
  y: number;
  w: number;
  h: number;
  /** for "add" circles: where the residual rail starts and where the sub-layer joins */
  join?: { fromY: number; joinX: number };
}
export type Anchors = Record<string, Anchor[]>;

export function makeRegistry() {
  const anchors: Anchors = {};
  const reg = (id: string, a: Anchor) => {
    (anchors[id] ??= []).push(a);
  };
  return { anchors, reg };
}

interface Props {
  anchors: Anchors;
  ids: string[];
  motion: boolean;
  speed: number;
  which: 'model' | 'block';
}

function Dot({ path, cx, cy, dur, motion }: { path: string; cx: number; cy: number; dur: string; motion: boolean }) {
  if (!motion) return <circle cx={cx} cy={cy} r={4.2} fill={SEL} stroke={INK} strokeWidth={1} />;
  return (
    <circle r={4.2} fill={SEL} stroke={INK} strokeWidth={1}>
      <animateMotion dur={dur} repeatCount="indefinite" path={path} calcMode="spline" keyTimes="0;1" keySplines="0.4 0 0.6 1" />
      <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.15;0.8;1" dur={dur} repeatCount="indefinite" />
    </circle>
  );
}

/** Amber rings around the active boxes, plus a dot that travels through them. Draws on top of a diagram. */
export default function FlowMarks({ anchors, ids, motion, speed, which }: Props) {
  const dur = `${(1.6 / speed).toFixed(2)}s`;
  const rects = ids.flatMap((id) => anchors[id] ?? []);
  return (
    <g className={`flow-marks flow-${which}`} pointerEvents="none">
      {rects.map((r, i) => (
        <rect
          key={'ring' + i}
          className={'flow-ring' + (motion ? ' pulse' : '')}
          x={r.x - 4}
          y={r.y - 4}
          width={r.w + 8}
          height={r.h + 8}
          rx={r.join ? r.w / 2 + 4 : 6}
          fill={SEL}
          fillOpacity={0.13}
          stroke={SEL}
          strokeWidth={2.4}
        />
      ))}
      {rects.map((r, i) => {
        const cx = r.x + r.w / 2;
        const cy = r.y + r.h / 2;
        if (r.join) {
          return (
            <g key={'dot' + i}>
              <Dot cx={cx} cy={cy + 14} dur={dur} motion={motion} path={`M ${cx} ${r.join.fromY} L ${cx} ${cy}`} />
              <Dot cx={cx + 14} cy={cy} dur={dur} motion={motion} path={`M ${r.join.joinX} ${cy} L ${cx} ${cy}`} />
            </g>
          );
        }
        return <Dot key={'dot' + i} cx={cx} cy={cy} dur={dur} motion={motion} path={`M ${cx} ${r.y + r.h + 16} L ${cx} ${r.y - 16}`} />;
      })}
    </g>
  );
}
