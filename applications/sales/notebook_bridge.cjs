"use strict";
// JSON stdin/stdout adapter for the Python walkthrough. No files are written.
const fs = require('node:fs');
const core = require('../../src/kikuchi.cjs');
const sales = require('./numerical_core.cjs');
const request = JSON.parse(fs.readFileSync(0, 'utf8'));
const {operation, dimension: d, coefficients, level, seed = 0} = request;
let result;
switch (operation) {
  case 'tensor':
    result = {coefficients: Array.from(sales.cumulants(sales.moments(request.codes, d), d))};
    break;
  case 'spectrum': {
    const {structure, op, lanczos} = core._internals;
    const shape = structure(d, level);
    const multiply = op(shape, coefficients);
    const eigen = lanczos(multiply, shape.D, 6, seed);
    const probe = Float64Array.from({length: shape.D}, (_, i) => Math.sin(i + 1));
    result = {
      values: eigen.values, vectors: eigen.vectors.map(v => Array.from(v)),
      residual: eigen.residual, steps: eigen.steps, dimension: shape.D,
      probe_product: Array.from(multiply(probe)),
    };
    break;
  }
  case 'recover': {
    const recovered = core.recoverDirection(coefficients, {dimension: d, level, seed});
    result = {direction: Array.from(recovered.direction), info: recovered.info};
    break;
  }
  case 'permutation':
    result = sales.permutation(request.codes, request.direction, d, request.count, seed);
    break;
  default:
    throw new Error('Unknown notebook operation: ' + operation);
}
process.stdout.write(JSON.stringify(result));
