export const INK = '#142033';
export const INK2 = '#4B5B70';
export const INK3 = '#7D8B9D';
export const HAIR = '#C9D1DA';
export const PAPER = '#F2F4F6';
export const PLATE = '#FBFCFD';
export const SEL = '#E8A400';

export const FONT_UI = "'Schibsted Grotesk', system-ui, sans-serif";
export const FONT_MONO = "'JetBrains Mono', ui-monospace, Menlo, monospace";

export const tint = (hex: string, a = 0.1) => {
  const n = Math.round(a * 255).toString(16).padStart(2, '0');
  return hex + n;
};

/** plain-language explanation of each tensor role, shown in the inspector */
export const ROLE_TEXT: Record<string, string> = {
  embed: 'Lookup table. Each token id selects one row, which becomes that token’s starting vector.',
  pos: 'Learned vector added to each token to tell the model where it sits in the sequence.',
  head: 'Projects the final hidden vector back to one score per vocabulary token (the logits).',
  norm: 'Scales each vector to a stable size before the next operation. Cheap, but present twice per block.',
  bias: 'Constant offset added after a projection.',
  q: 'Projects each token into query vectors: what each head is looking for.',
  k: 'Projects each token into key vectors: what each token offers to be matched against. Fewer KV heads than query heads is what makes grouped-query attention cheaper to cache.',
  v: 'Projects each token into value vectors: the content that gets passed on when a key matches.',
  o: 'Mixes the outputs of all heads back into a single vector of the model’s hidden size.',
  q_a: 'Low-rank “down” projection of queries into a small latent space (multi-head latent attention).',
  q_b: 'Expands the latent query back out into per-head queries, split into content and rotary parts.',
  kv_a: 'Compresses keys and values into one small latent vector plus a shared rotary key. This latent is all that gets cached.',
  kv_b: 'Expands the cached latent back into per-head keys and values when attention is computed.',
  gate: 'One of two parallel expansions of the MLP. Its activation decides how much of the other branch passes through.',
  up: 'Expands the hidden vector into the wider feed-forward space.',
  down: 'Projects the wide feed-forward activations back to the hidden size.',
  router: 'Scores every expert for each token. The top few scores choose which experts run.',
  e_gate: 'Gate branch inside each routed expert. Only the top-k experts run per token.',
  e_up: 'Up-projection inside each routed expert.',
  e_down: 'Down-projection inside each routed expert.',
  s_gate: 'Gate branch of the shared expert, which runs for every token.',
  s_up: 'Up-projection of the shared expert.',
  s_down: 'Down-projection of the shared expert.',
  s_router: 'Scalar gate that scales the shared expert’s output.',
};
