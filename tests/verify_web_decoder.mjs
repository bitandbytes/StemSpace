// Run the actual Orchestra Space vendored IAMF worker under Node, shimmed only
// for browser transport and WASM fetch. Input is the four-pulse position demo.
import {Worker} from 'node:worker_threads';
import {readFile} from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';

const [rootArg, file] = process.argv.slice(2);
if (!rootArg || !file) {
  console.error('Usage: node tests/verify_web_decoder.mjs /path/to/orchestra-space demo.iamf');
  process.exit(2);
}
const root = path.resolve(rootArg);
const source = `
const {parentPort,workerData}=require('node:worker_threads');
const fs=require('node:fs'), path=require('node:path'), vm=require('node:vm');
globalThis.self=globalThis;
self.location={href:'http://localhost/vendor/decoder_worker_bundle.js'};
self.addEventListener=(type,fn)=>{if(type==='message')parentPort.on('message',data=>fn({data}));};
globalThis.postMessage=data=>parentPort.postMessage(data);
globalThis.fetch=async url=>{
  if(!String(url).endsWith('wrapper.wasm'))throw new Error('Unexpected fetch '+url);
  return new Response(fs.readFileSync(path.join(workerData.root,'dist/vendor/decoder/wasm/wrapper.wasm')),
                      {headers:{'Content-Type':'application/wasm'}});
};
vm.runInThisContext(fs.readFileSync(path.join(workerData.root,'dist/vendor/decoder_worker_bundle.js'),'utf8'));
`;
const worker = new Worker(source, {eval: true, workerData: {root}, stdout: true, stderr: true});
let diagnostics = '';
for (const stream of [worker.stdout, worker.stderr]) stream.on('data', d => {
  diagnostics = (diagnostics + d).slice(-5000);
});
try {
  const input = await readFile(file);
  const result = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Decoder timed out')), 30000);
    const finish = (callback, value) => { clearTimeout(timer); callback(value); };
    worker.once('error', error => finish(reject, error));
    worker.on('message', data => {
      if (data.eventType === 'DECODER_CREATED') {
        worker.postMessage({eventType: 'DECODE_STANDALONE_IAMF', eventData: {
          iamfBytes: input.buffer.slice(input.byteOffset, input.byteOffset + input.byteLength),
          outputLayout: 14, channelOrdering: 0,
        }});
      }
      if (data.eventType === 'DECODE_DONE') finish(resolve, data.audioResult);
      if (data.eventType === 'ERROR') finish(reject, new Error(data.message));
    });
    worker.postMessage({eventType: 'CREATE_DECODER'});
  });
  assert.equal(result.numChannels, 2);
  assert.equal(result.sampleRate, 48000);
  const samples = new Int16Array(result.outputBuffers.reduce((n, block) => n + block.length, 0));
  let index = 0;
  for (const block of result.outputBuffers) {
    assert.ok(block instanceof Int16Array);
    samples.set(block, index);
    index += block.length;
  }
  assert.ok(Math.abs(samples.length / (2 * result.sampleRate) - 4) < 0.06);
  const rate = result.sampleRate;
  const energy = [];
  for (let slot = 0; slot < 4; slot++) {
    let left = 0, right = 0;
    for (let i = (slot * rate) + 6000; i < (slot * rate) + 43000; i++) {
      left += samples[2*i] ** 2;
      right += samples[2*i+1] ** 2;
    }
    assert.ok(left > 1e6 && right > 1e6, 'All four pulses should be audible');
    energy.push([left, right]);
  }
  assert.ok(energy[3][0] > energy[3][1], 'The left speaker pulse should favor the left ear');
  const rearBalance = energy[1][0] / energy[1][1];
  const highBalance = energy[2][0] / energy[2][1];
  assert.ok(Math.abs(rearBalance - highBalance) > 0.01, 'Changing elevation should change the binaural output');
  console.log(JSON.stringify({channels: result.numChannels, sampleRate: rate, pulses: energy.length,
                              rearBalance, highBalance, passed: true}));
} catch (error) {
  console.error(diagnostics);
  throw error;
} finally {
  await worker.terminate();
}
