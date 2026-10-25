export function fmtParams(n: number): string {
  const a = Math.abs(n);
  if (a >= 1e12) return (n / 1e12).toFixed(2) + 'T';
  if (a >= 1e9) return (n / 1e9).toFixed(a >= 1e11 ? 0 : a >= 1e10 ? 1 : 2) + 'B';
  if (a >= 1e6) return (n / 1e6).toFixed(a >= 1e8 ? 0 : 1) + 'M';
  if (a >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

export function fmtBytes(b: number): string {
  if (b >= 1e12) return (b / 1e12).toFixed(2) + ' TB';
  if (b >= 1e10) return (b / 1e9).toFixed(0) + ' GB';
  if (b >= 1e9) return (b / 1e9).toFixed(1) + ' GB';
  if (b >= 1e6) return (b / 1e6).toFixed(0) + ' MB';
  if (b >= 1e3) return (b / 1e3).toFixed(0) + ' KB';
  return b + ' B';
}

export function fmtCtx(n: number | null): string {
  if (!n) return 'n/a';
  if (n >= 1048576 && n % 1048576 === 0) return n / 1048576 + 'M';
  if (n >= 1024) return Math.round(n / 1024) + 'K';
  return String(n);
}

export const fmtInt = (n: number) => n.toLocaleString('en-US');
export const fmtPct = (x: number, digits = 1) => (x * 100).toFixed(digits) + '%';
export const shapeStr = (s: number[]) => s.join(' × ');

export function fmtCount(n: number | undefined): string {
  if (n === undefined) return 'n/a';
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + 'K';
  return String(n);
}

export function timeAgo(iso?: string): string {
  if (!iso) return '';
  const d = (Date.now() - new Date(iso).getTime()) / 86400000;
  if (d < 1) return 'today';
  if (d < 2) return 'yesterday';
  if (d < 45) return Math.round(d) + ' days ago';
  if (d < 700) return Math.round(d / 30) + ' months ago';
  return Math.round(d / 365) + ' years ago';
}
