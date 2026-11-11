"use strict";
const assert = require('node:assert/strict');
const {recoverDirection, quartetIndices} = require('../src/kikuchi.cjs');

const dimension = 12;
const coefficients = quartetIndices(dimension).map(s => s.every(i => i < 6) ? 0.12 : 0);
for (const level of [2, 4]) {
  const {direction, info} = recoverDirection(coefficients, {dimension, level, seed: 7});
  const alignment = direction.reduce((sum, value, i) => sum + (i < 6 ? value / Math.sqrt(6) : 0), 0);
  assert(Math.abs(alignment * alignment - 1) < 1e-10);
  console.log(JSON.stringify({level, squared_cosine: alignment * alignment, eigen_residual: info.residual, direction: Array.from(direction)}, null, 2));
}
