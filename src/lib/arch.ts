import type {
  Arch,
  AttnSpec,
  BreakdownRow,
  GroupKey,
  MoeSpec,
  Stage,
  TensorSpec,
} from './types';

const UNGATED = new Set(['gpt2', 'gpt_neox', 'gptj', 'opt', 'bloom', 'falcon', 'phi', 'gpt_bigcode']);
const TIED_BY_DEFAULT = new Set(['gpt2', 'gemma', 'gemma2', 'gemma3_text', 'opt', 'gpt_bigcode', 'falcon', 'bloom']);
/** families whose derived totals were checked against published parameter counts */
export const VERIFIED_TYPES = new Set(['llama', 'mistral', 'mixtral', 'qwen3_moe', 'deepseek_v3', 'gemma2', 'gpt2']);
const FOUR_NORMS = new Set(['gemma2', 'gemma3_text']);

export const prod = (s: number[]) => s.reduce((a, b) => a * b, 1);

function pick(c: any, ...keys: string[]): any {
  for (const k of keys) if (c[k] !== undefined && c[k] !== null) return c[k];
  return undefined;
}

function T(
  key: string,
  name: string,
  shape: number[],
  group: GroupKey,
  role: string,
  count = 1,
  activeCount = count,
): TensorSpec {
  return { key, name, shape, group, role, count, activeCount };
}

export class UnsupportedConfig extends Error {}

export function deriveArch(raw: any): Arch {
  const wrapped = raw?.text_config && (raw.text_config.hidden_size ?? raw.text_config.n_embd);
  const c: any = wrapped
    ? { ...raw.text_config, model_type: raw.text_config.model_type ?? raw.model_type }
    : raw;
  const notes: string[] = [];
  if (wrapped) notes.push('This is a multimodal config. Only the language model is drawn; vision and audio towers are left out.');

  const mt = String(c.model_type ?? 'unknown');
  const hidden = pick(c, 'hidden_size', 'n_embd', 'd_model');
  const layers = pick(c, 'num_hidden_layers', 'n_layer', 'num_layers');
  const heads = pick(c, 'num_attention_heads', 'n_head');
  const vocab = pick(c, 'vocab_size', 'padded_vocab_size');

  const ok = [hidden, layers, heads, vocab].every((n) => typeof n === 'number' && n > 0);
  if (!ok) {
    const ssm = /mamba|rwkv|xlstm|state/i.test(mt);
    throw new UnsupportedConfig(
      ssm
        ? `"${mt}" is a state-space or recurrent model. Cutaway only draws attention-based transformers so far.`
        : `This config doesn't have the usual transformer fields (hidden size, layers, attention heads, vocab). model_type "${mt}" isn't supported yet.`,
    );
  }

  const ungated = UNGATED.has(mt);
  const norm: Arch['norm'] = ungated || (c.layer_norm_eps !== undefined && c.rms_norm_eps === undefined) ? 'LayerNorm' : 'RMSNorm';
  const act: string = pick(c, 'hidden_act', 'hidden_activation', 'activation_function') ?? 'silu';

  // ---- attention -------------------------------------------------------
  const isMla = typeof c.kv_lora_rank === 'number';
  const kvHeads: number = pick(c, 'num_key_value_heads', 'n_head_kv', 'multi_query_group_num') ?? (c.multi_query === true ? 1 : heads);
  let attn: AttnSpec;
  if (isMla) {
    const qkNope = c.qk_nope_head_dim as number;
    const qkRope = c.qk_rope_head_dim as number;
    const vHead = c.v_head_dim as number;
    attn = {
      kind: 'MLA',
      heads,
      kvHeads: heads,
      headDim: qkNope + qkRope,
      qkNorm: false,
      qkvBias: false,
      oBias: false,
      mla: { qLora: typeof c.q_lora_rank === 'number' ? c.q_lora_rank : null, kvLora: c.kv_lora_rank, qkNope, qkRope, vHead },
    };
  } else {
    const headDim: number = c.head_dim ?? Math.round(hidden / heads);
    const qkvBias = c.attention_bias === true || mt === 'qwen2' || mt === 'qwen2_moe' || ungated;
    attn = {
      kind: kvHeads === heads ? 'MHA' : kvHeads === 1 ? 'MQA' : 'GQA',
      heads,
      kvHeads,
      headDim,
      qkNorm: mt.startsWith('qwen3') || mt === 'gemma3_text',
      qkvBias,
      oBias: c.attention_bias === true || ungated,
    };
  }

  // ---- mlp / moe -------------------------------------------------------
  const experts: number = pick(c, 'num_local_experts', 'num_experts', 'n_routed_experts') ?? 0;
  const denseFfn: number | undefined = pick(c, 'intermediate_size', 'n_inner', 'ffn_hidden_size') ?? (mt === 'gpt2' ? 4 * hidden : undefined);
  if (typeof denseFfn !== 'number' && experts === 0) {
    throw new UnsupportedConfig('The config has no feed-forward size (intermediate_size), so the MLP shape is unknown.');
  }
  const expertFfn: number = c.moe_intermediate_size ?? denseFfn ?? 0;
  let moe: MoeSpec | null = null;
  if (experts > 0) {
    const sharedCount: number = c.n_shared_experts ?? 0;
    const sharedFfn: number = c.shared_expert_intermediate_size ?? sharedCount * expertFfn;
    moe = {
      experts,
      topK: c.num_experts_per_tok ?? c.moe_top_k ?? 2,
      expertFfn,
      sharedFfn,
      routerBias: mt === 'deepseek_v3',
      sharedGate: typeof c.shared_expert_intermediate_size === 'number',
    };
  }
  const firstDense: number = c.first_k_dense_replace ?? 0;
  const mlpOnly: number[] = Array.isArray(c.mlp_only_layers) ? c.mlp_only_layers : [];
  const step: number = c.decoder_sparse_step ?? (typeof c.moe_layer_freq === 'number' ? c.moe_layer_freq : 1);
  const isMoeLayer = (l: number) => !!moe && l >= firstDense && !mlpOnly.includes(l) && (l + 1) % step === 0;

  const tied: boolean = c.tie_word_embeddings ?? TIED_BY_DEFAULT.has(mt);
  const ctx: number | null = pick(c, 'max_position_embeddings', 'n_positions', 'seq_length', 'max_sequence_length') ?? null;
  const normBias = norm === 'LayerNorm';
  const mlpBias = c.mlp_bias === true || (ungated && mt !== 'falcon') || (mt === 'falcon' && c.bias === true);

  // ---- tensors of one block -------------------------------------------
  const normTensors = (): TensorSpec[] => {
    const names = FOUR_NORMS.has(mt)
      ? ['input_layernorm', 'post_attention_layernorm', 'pre_feedforward_layernorm', 'post_feedforward_layernorm']
      : ['input_layernorm', 'post_attention_layernorm'];
    const out: TensorSpec[] = [];
    names.forEach((n, i) => {
      out.push(T(`norm.${i}`, `${n}.weight`, [hidden], 'norm', 'norm'));
      if (normBias) out.push(T(`norm.${i}.bias`, `${n}.bias`, [hidden], 'norm', 'norm'));
    });
    return out;
  };

  const attnTensors = (): TensorSpec[] => {
    const out: TensorSpec[] = [];
    if (attn.kind === 'MLA' && attn.mla) {
      const m = attn.mla;
      const qk = m.qkNope + m.qkRope;
      if (m.qLora) {
        out.push(T('attn.q_a', 'self_attn.q_a_proj.weight', [m.qLora, hidden], 'attn', 'q_a'));
        out.push(T('attn.q_a_norm', 'self_attn.q_a_layernorm.weight', [m.qLora], 'attn', 'norm'));
        out.push(T('attn.q_b', 'self_attn.q_b_proj.weight', [heads * qk, m.qLora], 'attn', 'q_b'));
      } else {
        out.push(T('attn.q', 'self_attn.q_proj.weight', [heads * qk, hidden], 'attn', 'q'));
      }
      out.push(T('attn.kv_a', 'self_attn.kv_a_proj_with_mqa.weight', [m.kvLora + m.qkRope, hidden], 'attn', 'kv_a'));
      out.push(T('attn.kv_a_norm', 'self_attn.kv_a_layernorm.weight', [m.kvLora], 'attn', 'norm'));
      out.push(T('attn.kv_b', 'self_attn.kv_b_proj.weight', [heads * (m.qkNope + m.vHead), m.kvLora], 'attn', 'kv_b'));
      out.push(T('attn.o', 'self_attn.o_proj.weight', [hidden, heads * m.vHead], 'attn', 'o'));
      return out;
    }
    out.push(T('attn.q', 'self_attn.q_proj.weight', [heads * attn.headDim, hidden], 'attn', 'q'));
    out.push(T('attn.k', 'self_attn.k_proj.weight', [kvHeads * attn.headDim, hidden], 'attn', 'k'));
    out.push(T('attn.v', 'self_attn.v_proj.weight', [kvHeads * attn.headDim, hidden], 'attn', 'v'));
    if (attn.qkvBias) {
      out.push(T('attn.q.bias', 'self_attn.q_proj.bias', [heads * attn.headDim], 'attn', 'bias'));
      out.push(T('attn.k.bias', 'self_attn.k_proj.bias', [kvHeads * attn.headDim], 'attn', 'bias'));
      out.push(T('attn.v.bias', 'self_attn.v_proj.bias', [kvHeads * attn.headDim], 'attn', 'bias'));
    }
    out.push(T('attn.o', 'self_attn.o_proj.weight', [hidden, heads * attn.headDim], 'attn', 'o'));
    if (attn.oBias) out.push(T('attn.o.bias', 'self_attn.o_proj.bias', [hidden], 'attn', 'bias'));
    if (attn.qkNorm) {
      out.push(T('attn.q_norm', 'self_attn.q_norm.weight', [attn.headDim], 'attn', 'norm'));
      out.push(T('attn.k_norm', 'self_attn.k_norm.weight', [attn.headDim], 'attn', 'norm'));
    }
    return out;
  };

  const denseMlp = (): TensorSpec[] => {
    const f = denseFfn as number;
    if (gatedMlp) {
      return [
        T('mlp.gate', 'mlp.gate_proj.weight', [f, hidden], 'dense', 'gate'),
        T('mlp.up', 'mlp.up_proj.weight', [f, hidden], 'dense', 'up'),
        T('mlp.down', 'mlp.down_proj.weight', [hidden, f], 'dense', 'down'),
      ];
    }
    const out = [
      T('mlp.up', 'mlp.fc_in.weight', [f, hidden], 'dense', 'up'),
      T('mlp.down', 'mlp.fc_out.weight', [hidden, f], 'dense', 'down'),
    ];
    if (mlpBias) {
      out.push(T('mlp.up.bias', 'mlp.fc_in.bias', [f], 'dense', 'bias'));
      out.push(T('mlp.down.bias', 'mlp.fc_out.bias', [hidden], 'dense', 'bias'));
    }
    return out;
  };
  const gatedMlp = !ungated;

  const moeMlp = (): TensorSpec[] => {
    const m = moe as MoeSpec;
    const out: TensorSpec[] = [T('moe.router', 'mlp.gate.weight', [m.experts, hidden], 'router', 'router')];
    if (m.routerBias) out.push(T('moe.router.bias', 'mlp.gate.e_score_correction_bias', [m.experts], 'router', 'bias'));
    out.push(T('moe.gate', 'mlp.experts.*.gate_proj.weight', [m.expertFfn, hidden], 'experts', 'e_gate', m.experts, m.topK));
    out.push(T('moe.up', 'mlp.experts.*.up_proj.weight', [m.expertFfn, hidden], 'experts', 'e_up', m.experts, m.topK));
    out.push(T('moe.down', 'mlp.experts.*.down_proj.weight', [hidden, m.expertFfn], 'experts', 'e_down', m.experts, m.topK));
    if (m.sharedFfn > 0) {
      out.push(T('shared.gate', 'mlp.shared_experts.gate_proj.weight', [m.sharedFfn, hidden], 'shared', 's_gate'));
      out.push(T('shared.up', 'mlp.shared_experts.up_proj.weight', [m.sharedFfn, hidden], 'shared', 's_up'));
      out.push(T('shared.down', 'mlp.shared_experts.down_proj.weight', [hidden, m.sharedFfn], 'shared', 's_down'));
      if (m.sharedGate) out.push(T('shared.router', 'mlp.shared_expert_gate.weight', [1, hidden], 'shared', 's_router'));
    }
    return out;
  };

  // ---- stages: runs of identical blocks ------------------------------
  const stages: Stage[] = [];
  for (let l = 0; l < layers; l++) {
    const kind: Stage['kind'] = isMoeLayer(l) ? 'moe' : 'dense';
    const last = stages[stages.length - 1];
    if (last && last.kind === kind) {
      last.to = l;
      last.count += 1;
      continue;
    }
    const tensors = [...attnTensors(), ...(kind === 'moe' ? moeMlp() : denseMlp()), ...normTensors()];
    stages.push({
      id: `s${stages.length}`,
      kind,
      from: l,
      to: l,
      count: 1,
      tensors,
      paramsPerBlock: tensors.reduce((a, t) => a + t.count * prod(t.shape), 0),
      activePerBlock: tensors.reduce((a, t) => a + t.activeCount * prod(t.shape), 0),
    });
  }

  // ---- embeddings and tail --------------------------------------------
  const embed: TensorSpec[] = [T('embed', 'embed_tokens.weight', [vocab, hidden], 'embed', 'embed')];
  if (mt === 'gpt2' || mt === 'opt') {
    embed.push(T('pos', 'position_embeddings.weight', [ctx ?? 2048, hidden], 'pos', 'pos'));
  }
  const tail: TensorSpec[] = [T('final_norm', 'norm.weight', [hidden], 'norm', 'norm')];
  if (normBias) tail.push(T('final_norm.bias', 'norm.bias', [hidden], 'norm', 'norm'));
  if (!tied) tail.push(T('lm_head', 'lm_head.weight', [vocab, hidden], 'head', 'head'));

  // ---- totals -----------------------------------------------------------
  const sum = (ts: TensorSpec[], active: boolean) => ts.reduce((a, t) => a + (active ? t.activeCount : t.count) * prod(t.shape), 0);
  const rows = new Map<GroupKey, BreakdownRow>();
  const add = (t: TensorSpec, mult: number) => {
    const r = rows.get(t.group) ?? { group: t.group, params: 0, active: 0 };
    r.params += t.count * prod(t.shape) * mult;
    r.active += t.activeCount * prod(t.shape) * mult;
    rows.set(t.group, r);
  };
  embed.forEach((t) => add(t, 1));
  tail.forEach((t) => add(t, 1));
  stages.forEach((s) => s.tensors.forEach((t) => add(t, s.count)));
  const order: GroupKey[] = ['embed', 'pos', 'attn', 'dense', 'router', 'experts', 'shared', 'norm', 'head'];
  const breakdown = order.filter((g) => rows.has(g)).map((g) => rows.get(g) as BreakdownRow);

  const totalParams = sum(embed, false) + sum(tail, false) + stages.reduce((a, s) => a + s.paramsPerBlock * s.count, 0);
  const activeParams = sum(embed, true) + sum(tail, true) + stages.reduce((a, s) => a + s.activePerBlock * s.count, 0);

  if (typeof c.num_nextn_predict_layers === 'number' && c.num_nextn_predict_layers > 0) {
    notes.push(`The config lists ${c.num_nextn_predict_layers} multi-token-prediction layer(s). They are in the checkpoint files but not drawn or counted here.`);
  }
  if (tied) notes.push('Input embeddings and the output head share one weight matrix, so it is counted once.');
  const sw: number | null = typeof c.sliding_window === 'number' ? c.sliding_window : null;
  if (sw) notes.push(`Sliding-window attention (${sw} tokens) is set. KV-cache estimates below assume full-length attention, which overstates memory for windowed layers.`);
  if (!VERIFIED_TYPES.has(mt)) notes.push(`model_type "${mt}" is drawn from the standard transformer layout. Compare the derived total with Hugging Face's reported count to see if anything is missing.`);

  const rs = pick(c, 'rope_scaling', 'rope_parameters');
  return {
    modelType: mt,
    verified: VERIFIED_TYPES.has(mt),
    vocab,
    hidden,
    layers,
    ctx,
    attn,
    moe,
    denseFfn: denseFfn ?? 0,
    act,
    gated: gatedMlp,
    tied,
    norm,
    ropeTheta: pick(c, 'rope_theta') ?? rs?.rope_theta ?? null,
    ropeScaling: rs ? String(rs.rope_type ?? rs.type ?? 'custom') : null,
    slidingWindow: sw,
    stages,
    embed,
    tail,
    totalParams,
    activeParams,
    breakdown,
    notes,
  };
}

export const GROUP_META: Record<GroupKey, { label: string; color: string }> = {
  embed: { label: 'Token embeddings', color: '#8C7B2A' },
  pos: { label: 'Position embeddings', color: '#A8943A' },
  attn: { label: 'Attention', color: '#2B59C3' },
  dense: { label: 'Dense MLP', color: '#0F8B8D' },
  router: { label: 'MoE router', color: '#B4306B' },
  experts: { label: 'MoE experts', color: '#D1651B' },
  shared: { label: 'Shared experts', color: '#7B4FB8' },
  norm: { label: 'Norms', color: '#6B7A8C' },
  head: { label: 'Output head', color: '#2F3E52' },
};
