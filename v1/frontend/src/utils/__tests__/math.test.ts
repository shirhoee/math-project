import { describe, it, expect } from 'vitest';
import {
  createRotationMatrix,
  determinant,
  orthogonalityError,
  createIntrinsics,
  invertIntrinsics,
  matrixMultiply,
  computeHomography,
  matrixToLatex,
} from '../math';

describe('Camera Math and Matrix Computations', () => {
  it('creates identity rotation for 0 degrees', () => {
    const R = createRotationMatrix(0, 0, 0);
    expect(R[0][0]).toBeCloseTo(1.0);
    expect(R[1][1]).toBeCloseTo(1.0);
    expect(R[2][2]).toBeCloseTo(1.0);
    expect(R[0][1]).toBeCloseTo(0.0);
    expect(R[1][0]).toBeCloseTo(0.0);
  });

  it('guarantees rotation matrix orthogonality (R * R^T = I)', () => {
    const R = createRotationMatrix(15, -10, 5);
    const err = orthogonalityError(R);
    expect(err).toBeLessThan(1e-6);
  });

  it('guarantees rotation matrix determinant is 1.0 (det(R) = 1)', () => {
    const R = createRotationMatrix(25, 12, 0);
    const det = determinant(R);
    expect(det).toBeCloseTo(1.0, 5);
  });

  it('correctly inverts camera intrinsics matrix K', () => {
    const K = createIntrinsics(500, 500, 320, 240);
    const K_inv = invertIntrinsics(K);
    const I = matrixMultiply(K, K_inv);

    expect(I[0][0]).toBeCloseTo(1.0);
    expect(I[1][1]).toBeCloseTo(1.0);
    expect(I[2][2]).toBeCloseTo(1.0);
    expect(I[0][1]).toBeCloseTo(0.0);
    expect(I[0][2]).toBeCloseTo(0.0);
  });

  it('computes plane-induced homography H_k', () => {
    const K = createIntrinsics(500, 500, 320, 240);
    const R = createRotationMatrix(5, 2, 0);
    const t: [number, number, number] = [0.05, 0.0, 0.0];
    const n: [number, number, number] = [0, 0, 1];
    const d = 2.0;

    const H = computeHomography(K, R, t, n, d);
    expect(H).toHaveLength(3);
    expect(H[0]).toHaveLength(3);
    expect(Number.isFinite(H[0][0])).toBe(true);
  });

  it('formats matrices as valid KaTeX LaTeX bmatrix strings', () => {
    const K = createIntrinsics(500, 500, 320, 240);
    const latex = matrixToLatex(K, 1);
    expect(latex).toContain('\\begin{bmatrix}');
    expect(latex).toContain('\\end{bmatrix}');
    expect(latex).toContain('500.0');
    expect(latex).toContain('320.0');
  });
});
