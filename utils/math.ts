import { Matrix, random, add, matrix, norm, divide, row, MathCollection } from 'mathjs';

export const generateSyntheticData = (
  nClusters: number,
  nPointsPerCluster: number,
  dimensions: number
): number[][] => {
  // Define random cluster centers
  const clusterCenters: Matrix[] = [];
  for (let i = 0; i < nClusters; i++) {
    clusterCenters.push(matrix(random([1, dimensions], 0, 1)) as Matrix);
  }

  // Initialize an empty array for storing the points
  let points: Matrix = matrix([]);

  // Generate points around the cluster centers
  for (const center of clusterCenters) {
    const noise: Matrix = matrix(random([nPointsPerCluster, dimensions], -0.1, 0.1)) as Matrix;
    const clusterPoints: Matrix = add(noise, center) as Matrix;
    points = points.size()[0] ? matrix([...points.toArray(), ...clusterPoints.toArray()] as MathCollection) : clusterPoints;
  }

  // Normalize the vectors
  const rows = points.size()[0];
  for (let i = 0; i < rows; i++) {
    const vector = matrix(row(points, i));
    const shitfuck = vector.toArray() as number[]
    const vectorNorm = norm(shitfuck[0]);
    const normalizedVector = divide(vector, vectorNorm) as Matrix;
    (normalizedVector.toArray() as number[][])[0].forEach((value, index) => {
      points.set([i, index], value);
    });
  }

  return points.toArray() as number[][];
};

// Utility function to generate a random number between min and max
function randomFloat(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

// Function to cap the point values between -1 and 1
function capPoint(point: number[]): number[] {
  return point.map(coord => Math.max(-1, Math.min(1, coord)));
}

// Main function to generate clusters
export function generateSyntheticDataFast(nClusters: number, nPointsPerCluster: number, dimensions: number): number[][] {
  const points: number[][] = [];

  for (let i = 0; i < nClusters; i++) {
    // Generate a random center for this cluster
    const center: number[] = [];
    for (let d = 0; d < dimensions; d++) {
      center.push(randomFloat(-1, 1));
    }

    // Generate the points for this cluster
    for (let j = 0; j < nPointsPerCluster; j++) {
      const point: number[] = [];
      for (let d = 0; d < dimensions; d++) {
        const offset = randomFloat(-0.1, 0.1);  // Adjust the range of the offset as needed
        point.push(center[d] + offset);
      }
      points.push(capPoint(point));
    }
  }

  return points;
}