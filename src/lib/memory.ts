import type { Arch } from './types';

export interface Precision {
  id: string;
  label: string;
  bits: number;
}

export const PRECISIONS: Precision[] = [
  { id: 'fp32', label: 'FP32', bits: 32 },
  { id: 'bf16', label: 'BF16 / FP16', bits: 16 },
  { id: 'int8', label: 'FP8 / INT8', bits: 8 },
  { id: 'int4', label: '4-bit', bits: 4.5 },
];

export const KV_PRECISIONS: Precision[] = [
  { id: 'kv16', label: 'FP16', bits: 16 },
  { id: 'kv8', label: 'FP8', bits: 8 },
];

export interface Gpu {
  name: string;
  vramGB: number;
  bandwidthGBs: number;
}

export const GPUS: Gpu[] = [
  { name: 'RTX 3060', vramGB: 12, bandwidthGBs: 360 },
  { name: 'RTX 4060 Ti 16 GB', vramGB: 16, bandwidthGBs: 288 },
  { name: 'RTX 3090', vramGB: 24, bandwidthGBs: 936 },
  { name: 'RTX 4090', vramGB: 24, bandwidthGBs: 1008 },
  { name: 'RTX 5090', vramGB: 32, bandwidthGBs: 1792 },
  { name: 'L40S', vramGB: 48, bandwidthGBs: 864 },
  { name: 'A100 80 GB', vramGB: 80, bandwidthGBs: 2039 },
  { name: 'H100 80 GB', vramGB: 80, bandwidthGBs: 3350 },
  { name: 'H200 141 GB', vramGB: 141, bandwidthGBs: 4800 },
  { name: 'MI300X', vramGB: 192, bandwidthGBs: 5300 },
];

export function weightBytes(a: Arch, bits: number): number {
  return (a.totalParams * bits) / 8;
}

/** bytes of KV cache for a single token */
export function kvBytesPerToken(a: Arch, kvBits: number): number {
  const bytes = kvBits / 8;
  if (a.attn.kind === 'MLA' && a.attn.mla) {
    // latent cache: compressed KV plus the shared rope key
    return a.layers * (a.attn.mla.kvLora + a.attn.mla.qkRope) * bytes;
  }
  return 2 * a.layers * a.attn.kvHeads * a.attn.headDim * bytes;
}

export interface MemoryPlan {
  weights: number;
  kv: number;
  overhead: number;
  total: number;
}

export function memoryPlan(a: Arch, bits: number, kvBits: number, ctx: number, batch: number): MemoryPlan {
  const weights = weightBytes(a, bits);
  const kv = kvBytesPerToken(a, kvBits) * ctx * batch;
  const overhead = Math.max(1e9, weights * 0.06);
  return { weights, kv, overhead, total: weights + kv + overhead };
}

/** upper bound for single-stream decoding: every active weight is read once per token */
export function decodeCeiling(a: Arch, bits: number, gpu: Gpu, count: number): number {
  const activeBytes = (a.activeParams * bits) / 8;
  return (gpu.bandwidthGBs * 1e9 * count) / activeBytes;
}

export function gpusNeeded(plan: MemoryPlan, gpu: Gpu): number {
  return Math.ceil(plan.total / (gpu.vramGB * 1e9));
}

/** forward-pass compute per generated token (two FLOPs per active weight) */
export const flopsPerToken = (a: Arch) => 2 * a.activeParams;
