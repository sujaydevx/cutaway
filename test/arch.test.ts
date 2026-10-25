import { deriveArch } from '../src/lib/arch';
import { SAMPLES } from '../src/lib/samples';
import { kvBytesPerToken } from '../src/lib/memory';

let failed = 0;
function check(name: string, ok: boolean, detail = '') {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
  if (!ok) failed++;
}
const near = (a: number, b: number, tol: number) => Math.abs(a - b) / b <= tol;

const by = (id: string) => deriveArch(SAMPLES.find((s) => s.id === id)!.config);

// exact totals published on the model pages
for (const s of SAMPLES) {
  if (!s.reportedTotal) continue;
  const a = deriveArch(s.config);
  check(`${s.short} total params`, a.totalParams === s.reportedTotal, `${a.totalParams} vs ${s.reportedTotal}`);
}

const mix = by('mistralai/Mixtral-8x7B-v0.1');
check('Mixtral active params ~12.9B', near(mix.activeParams, 12.88e9, 0.005), String(mix.activeParams));
check('Mixtral has one stage', mix.stages.length === 1 && mix.stages[0].count === 32);

const ds = by('deepseek-ai/DeepSeek-V3');
check('DeepSeek-V3 main-model params ~671B', near(ds.totalParams, 671e9, 0.005), String(ds.totalParams));
check('DeepSeek-V3 active ~37B', near(ds.activeParams, 37e9, 0.03), String(ds.activeParams));
check('DeepSeek-V3 stages 3 dense + 58 moe', ds.stages.length === 2 && ds.stages[0].count === 3 && ds.stages[1].count === 58);
check('DeepSeek-V3 is MLA', ds.attn.kind === 'MLA');
check('DeepSeek-V3 KV/token = 61*(512+64)*2', kvBytesPerToken(ds, 16) === 61 * 576 * 2);

const q = by('Qwen/Qwen3-30B-A3B');
check('Qwen3-30B-A3B active ~3.3B', near(q.activeParams, 3.3e9, 0.03), String(q.activeParams));

const ll = by('meta-llama/Llama-3.1-8B');
check('Llama KV/token = 2*32*8*128*2', kvBytesPerToken(ll, 16) === 2 * 32 * 8 * 128 * 2);
check('Llama is GQA', ll.attn.kind === 'GQA');

let threw = false;
try {
  deriveArch({ model_type: 'mamba', hidden_size: 768, num_hidden_layers: 24, vocab_size: 50280 });
} catch {
  threw = true;
}
check('state-space config is rejected with a clear error', threw);

console.log(failed ? `\n${failed} failed` : '\nall passed');
process.exit(failed ? 1 : 0);
