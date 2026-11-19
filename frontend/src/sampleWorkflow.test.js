import test from 'node:test';
import assert from 'node:assert/strict';
import {sampleIdText,testDraft,validateXraySelection} from './sampleWorkflow.js';

test('sample ID 1 is preserved as text in the saved draft',()=>{
  assert.equal(sampleIdText('1'),'1');
  assert.equal(sampleIdText(1),'1');
  assert.deepEqual(testDraft(' 1 ',new Date('2026-10-27T00:00:00Z')),{sampleId:'1',createdAt:'2026-10-27T00:00:00.000Z'});
});
test('blank and overlong IDs are rejected before creating a record',()=>{
  for(const value of ['', '   ', null, undefined, 'a'.repeat(121)])assert.throws(()=>testDraft(value));
});
test('JPEG selection tolerates missing browser MIME metadata',()=>{
  assert.doesNotThrow(()=>validateXraySelection({name:'sample-1.JPEG',type:'',size:1024}));
  assert.doesNotThrow(()=>validateXraySelection({name:'sample-1.png',type:'application/octet-stream',size:1024}));
  assert.doesNotThrow(()=>validateXraySelection({name:'sample-1.jpg',type:'image/jpeg',size:1024}));
});
test('unsupported, empty and oversized uploads are rejected',()=>{
  for(const file of [{name:'sample.txt',type:'',size:1024},{name:'sample.jpg',type:'text/plain',size:1024},{name:'sample.png',type:'image/png',size:0},{name:'sample.jpg',type:'image/jpeg',size:25*1024*1024+1}])assert.throws(()=>validateXraySelection(file));
});
