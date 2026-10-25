import type { Arch, Stage } from './types';

/**
 * One step of the "follow a token through the model" walkthrough.
 * `model` and `block` are anchor ids that the two diagrams register while they lay themselves out.
 */
export interface FlowStep {
  id: string;
  phase: 'model' | 'block';
  /** which stage the block diagram should be showing */
  stageIdx?: number;
  model: string[];
  block: string[];
  title: string;
  body: string;
  from?: string;
  to?: string;
}

/** dimensions are always shown exactly: researchers need 4096, not 4.1K */
const n = (x: number) => String(x);

function blockSteps(a: Arch, s: Stage, si: number): FlowStep[] {
  const d = a.hidden;
  const at = a.attn;
  const has = (k: string) => s.tensors.some((t) => t.key === k);
  const H = at.heads;
  const out: FlowStep[] = [];
  const add = (st: Omit<FlowStep, 'phase' | 'stageIdx' | 'model'>) =>
    out.push({ ...st, phase: 'block', stageIdx: si, model: [`stage:${si}`] });
  const stream = `[B, T, ${d}]`;

  add({
    id: `b${si}.in`,
    block: ['in'],
    title: 'Block input: the residual stream',
    body: `Each token is a vector of ${n(d)} numbers. This vector is the residual stream. It runs up the left rail past every sub-layer, and each sub-layer only adds a correction to it.`,
    to: stream,
  });
  add({
    id: `b${si}.norm0`,
    block: ['norm.0'],
    title: `${a.norm} before attention`,
    body: `Rescales every token vector to a steady size so the projections that follow see well-behaved numbers. The rail keeps the un-normalised copy.`,
    from: stream,
    to: stream,
  });

  let hd: number;
  let vd: number;
  if (at.kind === 'MLA' && at.mla) {
    const m = at.mla;
    hd = m.qkNope + m.qkRope;
    vd = m.vHead;
    add({
      id: `b${si}.mla.down`,
      block: ['mla.down'],
      title: 'Compress into small latents',
      body: `${m.qLora ? `Queries are squeezed from ${n(d)} to ${n(m.qLora)} numbers. ` : ''}Keys and values share one latent of ${n(m.kvLora)} numbers plus ${n(m.qkRope)} for position. This ${n(m.kvLora + m.qkRope)}-number latent is the only thing cached per token and layer.`,
      from: stream,
      to: `latent [B, T, ${n(m.kvLora + m.qkRope)}]`,
    });
    add({
      id: `b${si}.mla.norm`,
      block: ['mla.norm'],
      title: 'Normalise the latents',
      body: `A ${a.norm} on each compressed latent keeps the low-rank path numerically stable.`,
    });
    add({
      id: `b${si}.mla.up`,
      block: ['mla.up'],
      title: `Expand into ${H} heads`,
      body: `The latents are projected back up into per-head queries, keys and values. Each query and key has ${m.qkNope} content numbers plus ${m.qkRope} rotary numbers; each value has ${m.vHead}.`,
      to: `Q, K [B, ${H}, T, ${hd}]   V [B, ${H}, T, ${vd}]`,
    });
  } else {
    hd = at.headDim;
    vd = at.headDim;
    const kv = at.kvHeads;
    add({
      id: `b${si}.qkv`,
      block: ['qkv'],
      title: 'Project to queries, keys and values',
      body: `Three parallel linear layers turn each token vector into what it is looking for (Q), what it offers (K) and what it passes on (V). ${
        at.kind === 'MHA'
          ? `There are ${H} heads, each with its own Q, K and V of size ${hd}.`
          : at.kind === 'MQA'
            ? `${H} query heads all share a single key/value head.`
            : `${H} query heads share ${kv} key/value heads, so each K and V is read by ${H / kv} query heads. The cache only stores ${kv} of them.`
      }`,
      from: stream,
      to: `Q [B, ${H}, T, ${hd}]   K, V [B, ${kv}, T, ${hd}]`,
    });
    if (at.qkNorm) {
      add({
        id: `b${si}.qknorm`,
        block: ['qknorm'],
        title: 'Normalise queries and keys',
        body: `Each head's Q and K vectors (${hd} numbers) are rescaled on their own, which keeps the attention scores from growing too large.`,
      });
    }
    if (a.ropeTheta) {
      add({
        id: `b${si}.rope`,
        block: ['rope'],
        title: 'Rotary position encoding',
        body: `Q and K are rotated by an angle that depends on each token's position (base θ = ${n(a.ropeTheta)}${a.ropeScaling ? `, ${a.ropeScaling} scaling` : ''}). The dot product of two tokens then depends on how far apart they are, without adding any position vector.`,
      });
    }
  }

  add({
    id: `b${si}.sdpa`,
    block: ['sdpa'],
    title: 'Attention: look back at earlier tokens',
    body: `Every token compares its query with the keys of all earlier tokens: scores = Q·Kᵀ / √${hd}. Future positions are masked, softmax turns each row of scores into weights, and the weights mix the values. The score matrix is [B, ${H}, T, T], which is why long contexts get expensive.${a.slidingWindow ? ` Some layers only look back ${n(a.slidingWindow)} tokens.` : ''}`,
    to: `[B, T, ${n(H * vd)}]`,
  });
  add({
    id: `b${si}.o`,
    block: ['attn.o'],
    title: 'Merge the heads',
    body: `The ${H} head outputs are laid side by side (${n(H * vd)} numbers per token) and o_proj projects them back to ${n(d)}.`,
    from: `[B, T, ${n(H * vd)}]`,
    to: stream,
  });
  add({
    id: `b${si}.add1`,
    block: ['add1'],
    title: 'Add back to the residual stream',
    body: `The attention result is added to the vector that went around it on the rail. The layer only has to learn a correction, not rebuild the whole vector.`,
    to: stream,
  });
  add({
    id: `b${si}.norm1`,
    block: ['norm.1'],
    title: `${a.norm} before feed-forward`,
    body: `The same rescaling again, now ahead of the feed-forward part of the block.`,
    from: stream,
    to: stream,
  });

  if (s.kind === 'moe' && a.moe) {
    const m = a.moe;
    add({
      id: `b${si}.router`,
      block: ['router'],
      title: 'Router chooses experts',
      body: `A small linear layer (${n(d)} → ${n(m.experts)}) gives every expert a score for this token. The top ${m.topK} win and their scores become mixing weights.`,
      from: stream,
      to: `scores [B, T, ${n(m.experts)}]`,
    });
    add({
      id: `b${si}.experts`,
      block: m.sharedFfn ? ['experts', 'shared'] : ['experts'],
      title: `Run ${m.topK} of ${n(m.experts)} experts`,
      body: `Each chosen expert is a gated MLP (${n(d)} → ${n(m.expertFfn)} → ${n(d)}). The other ${n(m.experts - m.topK)} do no work for this token, which is why only ${Math.round((a.activeParams / a.totalParams) * 100)}% of the weights are used.${m.sharedFfn ? ` The shared expert (${n(m.sharedFfn)} wide) runs for every token alongside them.` : ''}`,
    });
    add({
      id: `b${si}.wsum`,
      block: ['wsum'],
      title: 'Weighted sum of expert outputs',
      body: `Each chosen expert's output is multiplied by its router weight and the results are added${m.sharedFfn ? ', together with the shared expert' : ''}.`,
      to: stream,
    });
  } else {
    const f = a.denseFfn;
    add({
      id: `b${si}.expand`,
      block: ['mlp.expand'],
      title: `Expand: ${n(d)} → ${n(f)}`,
      body: a.gated
        ? `Two parallel linear layers, gate_proj and up_proj, widen every token vector to ${n(f)} numbers.`
        : `fc_in widens every token vector from ${n(d)} to ${n(f)} numbers.`,
      from: stream,
      to: `[B, T, ${n(f)}]`,
    });
    add({
      id: `b${si}.act`,
      block: ['mlp.act'],
      title: a.gated ? `${a.act}(gate) ⊙ up` : `${a.act} activation`,
      body: a.gated
        ? `The gate branch goes through ${a.act}, then multiplies the up branch number by number. The gate decides how much of each feature gets through.`
        : `${a.act} is applied to every number. This non-linearity is what lets the feed-forward layer learn more than a straight line.`,
    });
    add({
      id: `b${si}.down`,
      block: ['mlp.down'],
      title: `Compress: ${n(f)} → ${n(d)}`,
      body: `${a.gated ? 'down_proj' : 'fc_out'} projects the wide vector back down to the hidden size.`,
      from: `[B, T, ${n(f)}]`,
      to: stream,
    });
  }

  add({
    id: `b${si}.add2`,
    block: ['add2'],
    title: 'Add back to the residual stream',
    body: `The feed-forward result joins the rail the same way the attention result did.`,
    to: stream,
  });
  add({
    id: `b${si}.out`,
    block: ['out'],
    title: 'Block output',
    body: `The updated stream leaves the block with the same shape it came in with. That is what lets blocks be stacked.`,
    to: stream,
  });
  return out;
}

export function buildFlow(a: Arch): FlowStep[] {
  const d = a.hidden;
  const stream = `[B, T, ${d}]`;
  const steps: FlowStep[] = [];

  steps.push({
    id: 'ids',
    phase: 'model',
    model: ['ids'],
    block: [],
    title: 'Token ids go in',
    body: `The tokenizer has already split the text into T tokens. Each token is one integer between 0 and ${n(a.vocab - 1)}. That list of integers is all the network receives.`,
    to: '[B, T]',
  });
  steps.push({
    id: 'embed',
    phase: 'model',
    model: ['embed'],
    block: [],
    title: 'Embedding lookup',
    body: `Each id picks one row of a ${n(a.vocab)} × ${n(d)} table. That row becomes the token's starting vector of ${n(d)} numbers.${
      a.embed.length > 1 ? ' A learned position vector is added so the model knows the order of tokens.' : a.ropeTheta ? ' Position is not added here; rotary encoding handles it later, inside attention.' : ''
    }`,
    from: '[B, T]',
    to: stream,
  });

  a.stages.forEach((s, si) => {
    const what = a.moe ? (s.kind === 'moe' ? 'MoE blocks' : 'Dense blocks') : 'Decoder blocks';
    const range = s.count === 1 ? `layer ${s.from}` : `layers ${s.from}–${s.to}`;
    steps.push({
      id: `s${si}`,
      phase: 'model',
      stageIdx: si,
      model: [`stage:${si}`],
      block: [],
      title: `${what}, ${range}`,
      body: `${s.count === 1 ? 'One block' : `${s.count} blocks with the same shape`} ${s.count === 1 ? 'processes' : 'process'} the vectors next. The diagram on the right opens up the first one.`,
      from: stream,
      to: stream,
    });
    steps.push(...blockSteps(a, s, si));
    if (s.count > 1) {
      steps.push({
        id: `s${si}.repeat`,
        phase: 'model',
        stageIdx: si,
        model: [`stage:${si}`],
        block: [],
        title: `Repeat ×${s.count}`,
        body: `What you just saw runs ${s.count} times in a row (${range}), each time with its own weights. Every block reads the previous block's output.`,
        from: stream,
        to: stream,
      });
    }
  });

  steps.push({
    id: 'final_norm',
    phase: 'model',
    model: ['final_norm'],
    block: [],
    title: `Final ${a.norm}`,
    body: `One last rescaling of every token vector before the vocabulary projection.`,
    from: stream,
    to: stream,
  });
  steps.push({
    id: 'head',
    phase: 'model',
    model: ['head'],
    block: [],
    title: 'Output head',
    body: `Multiplies each vector by a ${n(d)} × ${n(a.vocab)} matrix${a.tied ? ' (the embedding table again, reused)' : ''}. The result is one score, a logit, for every token in the vocabulary.`,
    from: stream,
    to: `[B, T, ${n(a.vocab)}]`,
  });
  steps.push({
    id: 'probs',
    phase: 'model',
    model: ['probs'],
    block: [],
    title: 'Next-token probabilities',
    body: `Softmax turns the scores at the last position into probabilities. The model picks or samples the next token from them, appends it to the input, and runs again.`,
    from: `[B, T, ${n(a.vocab)}]`,
    to: `[B, ${n(a.vocab)}] for the last position`,
  });
  return steps;
}
