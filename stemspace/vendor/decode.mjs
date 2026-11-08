// Use the unmodified AOMedia standalone IAMF WebAssembly worker from Orchestra
// Space. Only the worker transport and local WASM fetch are adapted for Node.
import { Worker } from 'node:worker_threads';
import { readFile, open } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const [input, output, layout] = process.argv.slice(2);
const outputLayout = Number(layout);
if (!input || !output || !Number.isInteger(outputLayout)) {
  console.error('Expected input.iamf, output.wav and a decoder layout ID.');
  process.exit(2);
}
const root = path.dirname(fileURLToPath(import.meta.url));
const script = `
const {parentPort,workerData}=require('node:worker_threads');
const fs=require('node:fs'), path=require('node:path'), vm=require('node:vm');
globalThis.self=globalThis;
self.location={href:'http://localhost/decoder_worker_bundle.js'};
self.addEventListener=(type,fn)=>{if(type==='message')parentPort.on('message',data=>fn({data}));};
globalThis.postMessage=data=>{
  const transfers=[...new Set((data.audioResult?.outputBuffers ?? []).map(b=>b.buffer))];
  parentPort.postMessage(data,transfers);
};
globalThis.fetch=async url=>{
  if(!String(url).endsWith('wrapper.wasm'))throw new Error('Unexpected decoder fetch');
  const bytes=fs.readFileSync(path.join(workerData.root,'decoder/wasm/wrapper.wasm'));
  return new Response(bytes,{headers:{'Content-Type':'application/wasm'}});
};
vm.runInThisContext(fs.readFileSync(path.join(workerData.root,'decoder_worker_bundle.js'),'utf8'));
`;
const worker = new Worker(script, {eval:true, workerData:{root}, stdout:true, stderr:true});
let diagnostics = '';
for (const stream of [worker.stdout, worker.stderr]) stream.on('data', chunk => {
  diagnostics = (diagnostics + chunk).slice(-2500);
});

try {
  const inputBytes = await readFile(input);
  const iamfBytes = inputBytes.buffer.slice(inputBytes.byteOffset, inputBytes.byteOffset + inputBytes.byteLength);
  const audio = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('IAMF decoding timed out')), 180000);
    const finish = (fn, value) => { clearTimeout(timer); fn(value); };
    worker.once('error', error => finish(reject, error));
    worker.once('exit', code => finish(reject, new Error('Decoder exited early: ' + code)));
    worker.on('message', data => {
      if (data.eventType === 'DECODER_CREATED') {
        worker.postMessage({eventType:'DECODE_STANDALONE_IAMF',
          eventData:{iamfBytes,outputLayout,channelOrdering:0}}, [iamfBytes]);
      } else if (data.eventType === 'ERROR') {
        finish(reject, new Error(data.message));
      } else if (data.eventType === 'DECODE_DONE') {
        finish(resolve, data.audioResult);
      }
    });
    worker.postMessage({eventType:'CREATE_DECODER'});
  });

  const {outputBuffers, sampleRate, numChannels} = audio;
  if (!Number.isInteger(sampleRate) || sampleRate < 8000 || sampleRate > 384000 ||
      !Number.isInteger(numChannels) || numChannels < 1 || numChannels > 32 ||
      !outputBuffers?.length || outputBuffers.some(b => !(b instanceof Int16Array) || b.length % numChannels)) {
    throw new Error('The decoder returned invalid PCM audio.');
  }
  const byteCount = outputBuffers.reduce((sum, b) => sum + b.byteLength, 0);
  if (!byteCount || byteCount > 0xffffffff - 36) throw new Error('Decoded audio exceeds the WAV size limit.');
  const header = Buffer.alloc(44);
  header.write('RIFF', 0);
  header.writeUInt32LE(byteCount + 36, 4);
  header.write('WAVEfmt ', 8);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(numChannels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(sampleRate * numChannels * 2, 28);
  header.writeUInt16LE(numChannels * 2, 32);
  header.writeUInt16LE(16, 34);
  header.write('data', 36);
  header.writeUInt32LE(byteCount, 40);
  const fd = await open(output, 'w');
  try {
    // Batch frames for disk I/O without allocating a second full decoded file.
    await fd.writeFile(header);
    const scratch = Buffer.alloc(1024 * 1024);
    let used = 0;
    for (const block of outputBuffers) {
      const bytes = Buffer.from(block.buffer, block.byteOffset, block.byteLength);
      for (let start = 0; start < bytes.length;) {
        const n = Math.min(scratch.length - used, bytes.length - start);
        bytes.copy(scratch, used, start, start + n);
        used += n;
        start += n;
        if (used === scratch.length) {
          await fd.writeFile(scratch);
          used = 0;
        }
      }
    }
    if (used) await fd.writeFile(scratch.subarray(0, used));
  } finally {
    await fd.close();
  }
  console.log(JSON.stringify({sampleRate, numChannels, outputLayout:audio.outputLayout}));
} catch (error) {
  console.error(error.message || String(error));
  if (diagnostics) console.error(diagnostics.trim());
  process.exitCode = 1;
} finally {
  await worker.terminate();
}
