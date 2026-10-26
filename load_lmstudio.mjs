import { resolve } from 'node:path';
import { access } from 'node:fs/promises';
import { parseArgs } from 'node:util';
import { LMStudioClient } from './.tools/lms-sdk/node_modules/@lmstudio/sdk/dist/index.mjs';

const { values } = parseArgs({ options: {
  model: { type: 'string', default: 'recursivesi/flint@q4_k_m' },
  adapter: { type: 'string', default: 'adapters/opening-pruned' },
  identifier: { type: 'string', default: 'flint-terse-original-q4' },
  context: { type: 'string', default: '65536' },
} });
const adapter = resolve(values.adapter, 'adapter.gguf');
const template = resolve(values.adapter, 'chat_template.jinja');
await access(adapter);
await access(template);

const client = new LMStudioClient();
try {
  const model = await client.llm.load(values.model, {
    identifier: values.identifier,
    config: {
      contextLength: Number(values.context), maxParallelPredictions: 1,
      speculativeDraftMtp: false, gpu: { ratio: 1 },
      llamaCppArgumentsOverride: {
        enabled: true, excludeAllConfig: false,
        disabledParameters: ['--chat-template-file'],
        overrideParameters: [
          { key: '--lora', value: adapter },
          { key: '--chat-template-file', value: template },
        ],
      },
    },
  });
  console.log(`Ready in LM Studio: ${model.identifier}`);
} finally {
  await client[Symbol.asyncDispose]();
}
