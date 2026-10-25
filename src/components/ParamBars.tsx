import type { Arch } from '../lib/types';
import { GROUP_META } from '../lib/arch';
import { fmtBytes, fmtParams, fmtPct } from '../lib/format';

export function CompositionBar({ arch, which, height = 22 }: { arch: Arch; which: 'params' | 'active'; height?: number }) {
  const total = which === 'params' ? arch.totalParams : arch.activeParams;
  return (
    <div className="bar" style={{ height }} role="img" aria-label="Parameter composition">
      {arch.breakdown.map((r) => {
        const v = r[which];
        if (!v) return null;
        return (
          <span
            key={r.group}
            title={`${GROUP_META[r.group].label}: ${fmtParams(v)} (${fmtPct(v / total)})`}
            style={{ width: `${(v / total) * 100}%`, background: GROUP_META[r.group].color }}
          />
        );
      })}
    </div>
  );
}

export function Legend({ arch }: { arch: Arch }) {
  return (
    <ul className="legend">
      {arch.breakdown.map((r) => (
        <li key={r.group}>
          <span className="swatch" style={{ background: GROUP_META[r.group].color }} />
          {GROUP_META[r.group].label}
        </li>
      ))}
    </ul>
  );
}

export default function ParamBars({ arch, bits }: { arch: Arch; bits: number }) {
  const isMoe = !!arch.moe;
  const sorted = [...arch.breakdown].sort((a, b) => b.params - a.params);
  const max = sorted[0]?.params ?? 1;
  return (
    <div className="params">
      <div className="bar-block">
        <div className="bar-head">
          <span>All weights</span>
          <span className="mono">{fmtParams(arch.totalParams)}</span>
        </div>
        <CompositionBar arch={arch} which="params" />
      </div>
      {isMoe && (
        <div className="bar-block" style={{ width: `${Math.max((arch.activeParams / arch.totalParams) * 100, 6)}%`, minWidth: 180 }}>
          <div className="bar-head">
            <span>Used for one token</span>
            <span className="mono">
              {fmtParams(arch.activeParams)} · {fmtPct(arch.activeParams / arch.totalParams, 0)}
            </span>
          </div>
          <CompositionBar arch={arch} which="active" />
        </div>
      )}
      <table className="table">
        <thead>
          <tr>
            <th>Component</th>
            <th className="num">Parameters</th>
            <th className="num">Share</th>
            {isMoe && <th className="num">Used per token</th>}
            <th className="num">Size</th>
            <th aria-label="relative size" />
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => (
            <tr key={r.group}>
              <td>
                <span className="swatch" style={{ background: GROUP_META[r.group].color }} />
                {GROUP_META[r.group].label}
              </td>
              <td className="num mono">{fmtParams(r.params)}</td>
              <td className="num mono">{fmtPct(r.params / arch.totalParams)}</td>
              {isMoe && <td className="num mono">{fmtParams(r.active)}</td>}
              <td className="num mono">{fmtBytes((r.params * bits) / 8)}</td>
              <td className="minibar">
                <span style={{ width: `${(r.params / max) * 100}%`, background: GROUP_META[r.group].color }} />
              </td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr>
            <td>Total</td>
            <td className="num mono">{fmtParams(arch.totalParams)}</td>
            <td className="num mono">100%</td>
            {isMoe && <td className="num mono">{fmtParams(arch.activeParams)}</td>}
            <td className="num mono">{fmtBytes((arch.totalParams * bits) / 8)}</td>
            <td />
          </tr>
        </tfoot>
      </table>
    </div>
  );
}
