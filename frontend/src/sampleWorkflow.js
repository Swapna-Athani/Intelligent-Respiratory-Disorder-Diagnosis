export function sampleIdText(value) {
  const text = String(value ?? '').trim();
  if (!text) throw new Error('Enter a patient/sample ID before uploading or analyzing a sample.');
  if (text.length > 120) throw new Error('Patient/sample ID must be at most 120 characters.');
  return text;
}

export function testDraft(value, now = new Date()) {
  return {sampleId: sampleIdText(value), createdAt: now.toISOString()};
}

export function validateXraySelection(file) {
  if (!file || !file.size) throw new Error('Select a non-empty chest X-ray image.');
  if (file.size > 25 * 1024 * 1024) throw new Error('X-ray images must be at most 25 MB.');
  const knownType = ['image/png', 'image/jpeg'].includes(file.type);
  const unspecifiedType = !file.type || file.type === 'application/octet-stream';
  if (!knownType && !(unspecifiedType && /\.(png|jpe?g)$/i.test(file.name))) {
    throw new Error('Select a PNG or JPEG chest X-ray image.');
  }
}
