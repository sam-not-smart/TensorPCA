"use strict";
const test = require('node:test');
const assert = require('node:assert/strict');
const {recoverDirection, quartetIndices} = require('../src/kikuchi.cjs');

test('recover a signed six-mode spike at both lifts and both tensor signs', () => {
  const d = 12;
  const truth = Array.from({length:d}, (_,i) => i < 6 ? (i % 2 ? -1 : 1) / Math.sqrt(6) : 0);
  for (const sign of [1, -1]) {
    const coefficients = quartetIndices(d).map(s => sign * 4.32 * s.reduce((p,i) => p * truth[i], 1));
    for (const level of [2, 4]) {
      const result = recoverDirection(coefficients, {dimension:d, level, seed:7});
      const cosine = result.direction.reduce((a,b,i) => a + b * truth[i], 0);
      assert(Math.abs(cosine * cosine - 1) < 1e-10);
      assert(result.info.residual < 1e-8);
      assert.equal(result.info.dimension, level === 2 ? 66 : 495);
    }
  }
});

test('reject undefined recovery problems and invalid input', () => {
  assert.throws(() => recoverDirection([], {dimension:3}), /dimension/);
  assert.throws(() => recoverDirection([], {dimension:12}), /coefficients/);
  assert.throws(() => recoverDirection(new Float64Array(495), {dimension:12}), /zero tensor/);
  const bad = new Float64Array(495); bad[0] = NaN;
  assert.throws(() => recoverDirection(bad, {dimension:12}), /finite/);
  const values = new Float64Array(495); values[0] = 1;
  assert.throws(() => recoverDirection(values, {dimension:12, level:1}), /level/);
});
