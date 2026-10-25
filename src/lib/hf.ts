import { deriveArch, prod, UnsupportedConfig } from './arch';
import type { Arch } from './types';
import { SAMPLES } from './samples';

export interface ModelInfo {
  id: string;
  author: string;
  name: string;
  downloads?: number;
  likes?: number;
  lastModified?: string;
  license?: string;
  pipeline?: string;
  library?: string;
  tags: string[];
  baseModels: { id: string; relation: string }[];
  gated: boolean;
  reportedTotal?: number;
  dtypeParams?: Record<string, number>;
  files: string[];
}

export interface Tokenizer {
  vocabSize?: number;
  bos?: string;
  eos?: string;
  pad?: string;
  unk?: string;
  chatTemplate?: string;
  maxLength?: number;
}

export interface FileTensor {
  name: string;
  dtype: string;
  shape: number[];
}

export interface Verification {
  status: 'match' | 'partial' | 'unavailable';
  matched: number;
  total: number;
  file?: string;
  reason?: string;
}

export interface Related {
  id: string;
  downloads?: number;
}

export interface LoadedModel {
  key: string;
  id: string;
  source: 'hub' | 'sample' | 'pasted';
  info: ModelInfo;
  config: any;
  tokenizer?: Tokenizer;
  arch?: Arch;
  archError?: string;
  tensors?: FileTensor[];
  tensorFile?: string;
  verification?: Verification;
  quantized?: Related[];
  finetunes?: Related[];
}

export class HubError extends Error {
  constructor(
    message: string,
    public kind: 'notfound' | 'gated' | 'noconfig' | 'network' | 'invalid',
  ) {
    super(message);
  }
}

export function parseModelId(input: string): string | null {
  let s = input.trim();
  if (!s) return null;
  s = s.replace(/^https?:\/\/(www\.)?(huggingface\.co|hf\.co)\//i, '');
  s = s.split(/[?#]/)[0].replace(/^\/+|\/+$/g, '');
  const parts = s.split('/');
  if (parts[0] === 'models') parts.shift();
  if (parts.length < 2) return null;
  const [org, name] = parts;
  if (!/^[\w.-]+$/.test(org) || !/^[\w.-]+$/.test(name)) return null;
  return `${org}/${name}`;
}

const HUB = 'https://huggingface.co';

async function getJson(url: string, signal?: AbortSignal): Promise<any> {
  let res: Response;
  try {
    res = await fetch(url, { signal });
  } catch {
    throw new HubError('Could not reach huggingface.co. Check your connection, or paste the config.json instead.', 'network');
  }
  if (res.status === 404) throw new HubError('not found', 'notfound');
  if (res.status === 401 || res.status === 403) throw new HubError('gated', 'gated');
  if (!res.ok) throw new HubError(`Hugging Face answered ${res.status}.`, 'network');
  return res.json();
}

function toInfo(id: string, j: any): ModelInfo {
  const tags: string[] = Array.isArray(j.tags) ? j.tags : [];
  const license: string | undefined = j.cardData?.license ?? tags.find((t) => t.startsWith('license:'))?.slice(8);
  const baseModels: ModelInfo['baseModels'] = [];
  for (const t of tags) {
    const m = /^base_model:(?:(finetune|quantized|adapter|merge):)?(.+)$/.exec(t);
    if (m) baseModels.push({ id: m[2], relation: m[1] ?? 'base' });
  }
  if (!baseModels.length && j.cardData?.base_model) {
    const b = j.cardData.base_model;
    (Array.isArray(b) ? b : [b]).forEach((x: string) => baseModels.push({ id: String(x), relation: 'base' }));
  }
  const [author, name] = id.split('/');
  return {
    id,
    author,
    name,
    downloads: j.downloads,
    likes: j.likes,
    lastModified: j.lastModified,
    license,
    pipeline: j.pipeline_tag,
    library: j.library_name,
    tags,
    baseModels,
    gated: !!j.gated,
    reportedTotal: j.safetensors?.total,
    dtypeParams: j.safetensors?.parameters,
    files: Array.isArray(j.siblings) ? j.siblings.map((s: any) => s.rfilename) : [],
  };
}

function toTokenizer(j: any): Tokenizer {
  const tok = (v: any): string | undefined => (typeof v === 'string' ? v : v?.content);
  const tpl = typeof j.chat_template === 'string' ? j.chat_template : Array.isArray(j.chat_template) ? j.chat_template[0]?.template : undefined;
  return {
    bos: tok(j.bos_token),
    eos: tok(j.eos_token),
    pad: tok(j.pad_token),
    unk: tok(j.unk_token),
    chatTemplate: tpl,
    maxLength: typeof j.model_max_length === 'number' && j.model_max_length < 1e12 ? j.model_max_length : undefined,
  };
}

async function readHeader(url: string, signal?: AbortSignal): Promise<FileTensor[]> {
  const first = await fetch(url, { headers: { Range: 'bytes=0-7' }, signal });
  if (!first.ok) throw new Error(String(first.status));
  const buf = new DataView(await first.arrayBuffer());
  const n = Number(buf.getBigUint64(0, true));
  if (!n || n > 64e6) throw new Error('bad header');
  const second = await fetch(url, { headers: { Range: `bytes=8-${7 + n}` }, signal });
  if (!second.ok) throw new Error(String(second.status));
  const meta = JSON.parse(await second.text());
  return Object.entries(meta)
    .filter(([k]) => k !== '__metadata__')
    .map(([name, v]: [string, any]) => ({ name, dtype: v.dtype, shape: v.shape }));
}

async function loadTensors(id: string, files: string[], signal?: AbortSignal): Promise<{ tensors?: FileTensor[]; file?: string }> {
  try {
    let file: string | undefined = files.find((f) => f === 'model.safetensors');
    if (!file) {
      const index = files.find((f) => f === 'model.safetensors.index.json');
      if (index) {
        const j = await getJson(`${HUB}/${id}/resolve/main/${index}`, signal);
        const shards = Array.from(new Set(Object.values(j.weight_map as Record<string, string>))).sort();
        file = shards[0];
      } else {
        file = files.filter((f) => f.endsWith('.safetensors')).sort()[0];
      }
    }
    if (!file) return {};
    return { tensors: await readHeader(`${HUB}/${id}/resolve/main/${file}`, signal), file };
  } catch {
    return {};
  }
}

const LAYER0 = /(?:^|\.)(?:layers|h|blocks|layer)\.0\./;

export function verifyLayerZero(arch: Arch, tensors: FileTensor[] | undefined, file?: string): Verification {
  if (!tensors || !tensors.length) {
    return { status: 'unavailable', matched: 0, total: 0, reason: 'Could not read the weight files from your browser.' };
  }
  if (tensors.some((t) => /qweight|\.scales|\.qzeros/.test(t.name))) {
    return { status: 'unavailable', matched: 0, total: 0, file, reason: 'Quantized checkpoint: stored tensors differ from the full-precision layout.' };
  }
  const actual = new Map<string, number>();
  tensors
    .filter((t) => LAYER0.test(t.name))
    .forEach((t) => actual.set(t.shape.join('x'), (actual.get(t.shape.join('x')) ?? 0) + 1));
  if (!actual.size) {
    return { status: 'unavailable', matched: 0, total: 0, file, reason: 'Layer 0 is not in the first weight file.' };
  }
  const derived = new Map<string, number>();
  for (const t of arch.stages[0].tensors) {
    const k = t.shape.join('x');
    derived.set(k, (derived.get(k) ?? 0) + t.count);
  }
  let matched = 0;
  let total = 0;
  derived.forEach((n, k) => {
    total += n;
    matched += Math.min(n, actual.get(k) ?? 0);
  });
  return { status: matched / total >= 0.8 ? 'match' : 'partial', matched, total, file };
}

async function relatives(id: string, rel: 'quantized' | 'finetune', signal?: AbortSignal): Promise<Related[]> {
  try {
    const j = await getJson(`${HUB}/api/models?filter=base_model:${rel}:${id}&sort=downloads&limit=6`, signal);
    return (j as any[]).map((m) => ({ id: m.id ?? m.modelId, downloads: m.downloads }));
  } catch {
    return [];
  }
}

function build(key: string, id: string, source: LoadedModel['source'], info: ModelInfo, config: any): LoadedModel {
  const m: LoadedModel = { key, id, source, info, config };
  try {
    m.arch = deriveArch(config);
  } catch (e) {
    m.archError = e instanceof UnsupportedConfig ? e.message : 'Could not read this config: ' + (e as Error).message;
  }
  return m;
}

export async function loadFromHub(id: string, onStep: (s: string) => void, signal?: AbortSignal): Promise<LoadedModel> {
  onStep('Reading model page');
  let infoJson: any;
  try {
    infoJson = await getJson(`${HUB}/api/models/${id}`, signal);
  } catch (e) {
    if (e instanceof HubError && e.kind === 'notfound') throw new HubError(`No model called "${id}" on Hugging Face. Check the spelling of the URL.`, 'notfound');
    if (e instanceof HubError && e.kind === 'gated') throw new HubError(`"${id}" is private. Cutaway can only read public repos.`, 'gated');
    throw e;
  }
  const info = toInfo(id, infoJson);

  onStep('Reading config.json');
  let config: any;
  try {
    config = await getJson(`${HUB}/${id}/resolve/main/config.json`, signal);
  } catch (e) {
    if (e instanceof HubError && e.kind === 'notfound') {
      throw new HubError(`"${id}" has no config.json. It may be a GGUF-only or adapter-only repo. Try its base model, or paste a config.json.`, 'noconfig');
    }
    if (e instanceof HubError && e.kind === 'gated') {
      throw new HubError(`"${id}" is gated, so its files need a Hugging Face login. Open the repo, accept the terms, and paste its config.json here.`, 'gated');
    }
    throw e;
  }

  const model = build(`hub:${id}`, id, 'hub', info, config);

  onStep('Reading tokenizer and weight headers');
  const [tok, tens, quantized, finetunes] = await Promise.all([
    getJson(`${HUB}/${id}/resolve/main/tokenizer_config.json`, signal).then(toTokenizer).catch(() => undefined),
    model.arch ? loadTensors(id, info.files, signal) : Promise.resolve({} as { tensors?: FileTensor[]; file?: string }),
    relatives(id, 'quantized', signal),
    relatives(id, 'finetune', signal),
  ]);
  model.tokenizer = tok;
  model.tensors = tens.tensors;
  model.tensorFile = tens.file;
  model.quantized = quantized;
  model.finetunes = finetunes;
  if (model.arch) model.verification = verifyLayerZero(model.arch, tens.tensors, tens.file);
  return model;
}

export function loadPasted(text: string, label: string): LoadedModel {
  let config: any;
  try {
    config = JSON.parse(text);
  } catch {
    throw new HubError('That is not valid JSON. Paste the full contents of config.json.', 'invalid');
  }
  const id = label.trim() || 'pasted/config';
  const [author, name] = id.includes('/') ? id.split('/') : ['pasted', id];
  const info: ModelInfo = { id, author, name, tags: [], baseModels: [], gated: false, files: [] };
  return build(`pasted:${id}:${Date.now()}`, id, 'pasted', info, config);
}

export function loadSample(sampleId: string): LoadedModel {
  const s = SAMPLES.find((x) => x.id === sampleId) as (typeof SAMPLES)[number];
  const [author, name] = s.id.split('/');
  const info: ModelInfo = {
    id: s.id,
    author,
    name,
    license: s.license,
    pipeline: s.pipeline,
    tags: [],
    baseModels: [],
    gated: false,
    reportedTotal: s.reportedTotal,
    files: [],
  };
  return build(`sample:${s.id}`, s.id, 'sample', info, s.config);
}

export { prod };
