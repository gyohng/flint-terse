"""Fetch OpenClaw's workspace conditioning and apply it to local replay tasks."""

import argparse
import json
import subprocess
from pathlib import Path

from corpus import read_rows


# Render on the installed OpenClaw version. No agent turn or tool is executed.
SNAPSHOT = r"""
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const root = path.join(os.homedir(), '.npm-global/lib/node_modules/openclaw');
const dist = path.join(root, 'dist');
async function exported(prefix, symbol) {
  for (const name of fs.readdirSync(dist).filter(n => n.startsWith(prefix) && n.endsWith('.mjs'))) {
    const file = path.join(dist, name);
    const source = fs.readFileSync(file, 'utf8');
    const exports = source.slice(source.lastIndexOf('\nexport {'));
    const match = exports.match(new RegExp('\\b' + symbol + ' as (\\w+)\\b'));
    if (match) return (await import(pathToFileURL(file)))[match[1]];
  }
  throw new Error('OpenClaw export not found: ' + symbol);
}
const config = JSON.parse(fs.readFileSync(path.join(os.homedir(), '.openclaw/openclaw.json')));
const workspace = config.agents.entries?.main?.workspace ?? config.agents.defaults.workspace;
const contextFiles = ['AGENTS.md', 'SOUL.md'].map(name => ({
  path: path.join(workspace, name), content: fs.readFileSync(path.join(workspace, name), 'utf8')
}));
const tools = [];
for (const symbol of ['execTool', 'processTool']) {
  const tool = await exported('bash-tools-', symbol);
  tools.push({type: 'function', function: {
    name: tool.name, description: tool.description, parameters: tool.parameters
  }});
}
const render = await exported('system-prompt-', 'buildAgentSystemPrompt');
const system = render({
  workspaceDir: workspace, contextFiles, toolNames: tools.map(t => t.function.name),
  reasoningLevel: config.agents.defaults.reasoningDefault ?? 'off',
  runtimeInfo: {agentId: 'main', host: os.hostname(), os: os.platform(), model: 'lmstudio/recursivesi/flint'}
});
console.log(JSON.stringify({
  version: JSON.parse(fs.readFileSync(path.join(root, 'package.json'))).version,
  system, tools
}));
process.exit(0);
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sync = sub.add_parser("sync", help="Read remote instructions and tool schemas; no remote edits")
    sync.add_argument("--host", default="user@norda-minipc1.tail3e95b7.ts.net")
    sync.add_argument("--output", default="runs/openclaw-context.json")
    condition = sub.add_parser("condition", help="Add the captured system prompt and tools to tasks")
    condition.add_argument("input")
    condition.add_argument("--context", default="runs/openclaw-context.json")
    condition.add_argument("--output", required=True)
    args = p.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "sync":
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.host, "node --input-type=module"],
            input=SNAPSHOT, text=True, capture_output=True, check=True, timeout=60,
        )
        context = json.loads(result.stdout)
        output.write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n")
        print(f"OpenClaw {context['version']}: {len(context['system'])} prompt characters; exec/process schemas → {output}")
    else:
        context = json.loads(Path(args.context).read_text())
        rows = read_rows(args.input)
        for row in rows:
            if row.get("tools") or any(m["role"] in ("system", "developer") for m in row["messages"]):
                raise ValueError(f"{row['id']}: task already contains conditioning")
            row["messages"].insert(0, {"role": "system", "content": context["system"]})
            row["tools"] = context["tools"]
        with output.open("x") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{len(rows)} conditioned tasks → {output}")


if __name__ == "__main__":
    main()
