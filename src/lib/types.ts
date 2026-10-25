export type GroupKey =
  | 'embed'
  | 'pos'
  | 'attn'
  | 'dense'
  | 'router'
  | 'experts'
  | 'shared'
  | 'norm'
  | 'head';

export interface TensorSpec {
  /** unique inside a block, e.g. "attn.q_proj" */
  key: string;
  /** name as it usually appears in the checkpoint, relative to the layer */
  name: string;
  /** PyTorch weight shape, [out, in] for linear layers */
  shape: number[];
  /** how many copies exist per block (e.g. number of experts) */
  count: number;
  /** how many copies are used for one token (top-k experts) */
  activeCount: number;
  group: GroupKey;
  role: string;
}

export interface AttnSpec {
  kind: 'MHA' | 'GQA' | 'MQA' | 'MLA';
  heads: number;
  kvHeads: number;
  headDim: number;
  qkNorm: boolean;
  qkvBias: boolean;
  oBias: boolean;
  mla?: {
    qLora: number | null;
    kvLora: number;
    qkNope: number;
    qkRope: number;
    vHead: number;
  };
}

export interface MoeSpec {
  experts: number;
  topK: number;
  expertFfn: number;
  sharedFfn: number;
  routerBias: boolean;
  sharedGate: boolean;
}

export interface Stage {
  id: string;
  kind: 'dense' | 'moe';
  from: number;
  to: number;
  count: number;
  tensors: TensorSpec[];
  paramsPerBlock: number;
  activePerBlock: number;
}

export interface BreakdownRow {
  group: GroupKey;
  params: number;
  active: number;
}

export interface Arch {
  modelType: string;
  verified: boolean;
  vocab: number;
  hidden: number;
  layers: number;
  ctx: number | null;
  attn: AttnSpec;
  moe: MoeSpec | null;
  denseFfn: number;
  act: string;
  gated: boolean;
  tied: boolean;
  norm: 'RMSNorm' | 'LayerNorm';
  ropeTheta: number | null;
  ropeScaling: string | null;
  slidingWindow: number | null;
  stages: Stage[];
  embed: TensorSpec[];
  tail: TensorSpec[];
  totalParams: number;
  activeParams: number;
  breakdown: BreakdownRow[];
  notes: string[];
}
