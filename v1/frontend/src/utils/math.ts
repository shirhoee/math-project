export type Matrix3x3 = [
  [number, number, number],
  [number, number, number],
  [number, number, number],
];

export function degToRad(deg: number): number {
  return (deg * Math.PI) / 180;
}

export function matrixMultiply(A: Matrix3x3, B: Matrix3x3): Matrix3x3 {
  const result: Matrix3x3 = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let sum = 0;
      for (let k = 0; k < 3; k++) {
        sum += A[i][k] * B[k][j];
      }
      result[i][j] = sum;
    }
  }
  return result;
}

export function transpose(A: Matrix3x3): Matrix3x3 {
  return [
    [A[0][0], A[1][0], A[2][0]],
    [A[0][1], A[1][1], A[2][1]],
    [A[0][2], A[1][2], A[2][2]],
  ];
}

export function determinant(A: Matrix3x3): number {
  return (
    A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
    A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
    A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
  );
}

export function createRotationMatrix(yawDeg: number, pitchDeg: number, rollDeg = 0): Matrix3x3 {
  const y = degToRad(yawDeg);
  const p = degToRad(pitchDeg);
  const r = degToRad(rollDeg);

  const cy = Math.cos(y), sy = Math.sin(y);
  const cp = Math.cos(p), sp = Math.sin(p);
  const cr = Math.cos(r), sr = Math.sin(r);

  // Ry * Rx * Rz rotation
  // Ry: yaw
  const Ry: Matrix3x3 = [
    [cy, 0, sy],
    [0, 1, 0],
    [-sy, 0, cy],
  ];

  // Rx: pitch
  const Rx: Matrix3x3 = [
    [1, 0, 0],
    [0, cp, -sp],
    [0, sp, cp],
  ];

  // Rz: roll
  const Rz: Matrix3x3 = [
    [cr, -sr, 0],
    [sr, cr, 0],
    [0, 0, 1],
  ];

  return matrixMultiply(matrixMultiply(Ry, Rx), Rz);
}

export function orthogonalityError(R: Matrix3x3): number {
  const Rt = transpose(R);
  const RRt = matrixMultiply(R, Rt);
  let sumSq = 0;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const target = i === j ? 1 : 0;
      const diff = RRt[i][j] - target;
      sumSq += diff * diff;
    }
  }
  return Math.sqrt(sumSq);
}

export function createIntrinsics(fx: number, fy: number, cx: number, cy: number): Matrix3x3 {
  return [
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1],
  ];
}

export function invertIntrinsics(K: Matrix3x3): Matrix3x3 {
  const fx = K[0][0];
  const fy = K[1][1];
  const cx = K[0][2];
  const cy = K[1][2];

  return [
    [1 / fx, 0, -cx / fx],
    [0, 1 / fy, -cy / fy],
    [0, 0, 1],
  ];
}

export function computeHomography(
  K: Matrix3x3,
  R: Matrix3x3,
  t: [number, number, number],
  n: [number, number, number],
  d: number
): Matrix3x3 {
  // H = K * (R - (t * n^T) / d) * K_inv
  const K_inv = invertIntrinsics(K);

  const R_prime: Matrix3x3 = [
    [R[0][0] - (t[0] * n[0]) / d, R[0][1] - (t[0] * n[1]) / d, R[0][2] - (t[0] * n[2]) / d],
    [R[1][0] - (t[1] * n[0]) / d, R[1][1] - (t[1] * n[1]) / d, R[1][2] - (t[1] * n[2]) / d],
    [R[2][0] - (t[2] * n[0]) / d, R[2][1] - (t[2] * n[1]) / d, R[2][2] - (t[2] * n[2]) / d],
  ];

  return matrixMultiply(matrixMultiply(K, R_prime), K_inv);
}

export function matrixToLatex(matrix: Matrix3x3, precision = 3): string {
  const rows = matrix.map((row) =>
    row.map((val) => (Math.abs(val) < 1e-6 ? '0' : val.toFixed(precision))).join(' & ')
  );
  return `\\begin{bmatrix} ${rows.join(' \\\\ ')} \\end{bmatrix}`;
}
