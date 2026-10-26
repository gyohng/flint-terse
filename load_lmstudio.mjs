import { resolve } from 'node:path';
import { access, readFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { parseArgs } from 'node:util';
import { LMStudioClient } from './.tools/lms-sdk/node_modules/@lmstudio/sdk/dist/index.mjs';

const { values } = parseArgs({ options: {
  model: { type: 'string' },
  adapter: { type: 'string', default: 'release/flint-terse' },
  identifier: { type: 'string', default: 'flint-terse' },
  context: { type: 'string', default: '65536' },
  base: { type: 'boolean', default: false },
} });
const adapter = resolve(values.adapter, 'adapter.gguf');
const template = resolve(values.adapter, 'chat_template.jinja');
if (!values.base) await access(adapter);
await access(template);
let modelKey = values.model;
if (!modelKey) {
  const config = JSON.parse(await readFile(resolve(values.adapter, 'adapter_config.json'), 'utf8'));
  const listing = JSON.parse(execFileSync('lms', ['ls', '--variants', '--json'], { encoding: 'utf8' }));
  modelKey = listing.flatMap(entry => entry.variants ?? [entry]).find(model =>
    model.indexedModelIdentifier?.endsWith(`/${config.base_model.filename}`))?.modelKey;
  if (!modelKey) throw new Error(`Import ${config.base_model.filename} into LM Studio first, or specify --model.`);
}

const client = new LMStudioClient();
try {
  const model = await client.llm.load(modelKey, {
    identifier: values.identifier,
    config: {
      contextLength: Number(values.context), maxParallelPredictions: 1,
      speculativeDraftMtp: false, gpu: { ratio: 1 },
      llamaCppArgumentsOverride: {
        enabled: true, excludeAllConfig: false,
        disabledParameters: ['--chat-template-file'],
        overrideParameters: [
          ...values.base ? [] : [{ key: '--lora', value: adapter }],
          { key: '--chat-template-file', value: template },
        ],
      },
    },
  });
  console.log(`Ready in LM Studio: ${model.identifier}`);
} finally {
  await client[Symbol.asyncDispose]();
}
