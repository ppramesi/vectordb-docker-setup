import { generateSyntheticDataFast } from "../utils/math";

const max = 5;
const clusterNum = 100;
const dimensions = 1000;
const pointNums = 100;

test("Generate synthetic data fast", () => {
  for(let i = 0; i < max; i++){
    const startTime = performance.now();
    const synthData = generateSyntheticDataFast(clusterNum, pointNums, dimensions); // 100 clusters, pointNums points per cluster, 10 dimensions
    const endTime = performance.now();
    const timeElapsed = endTime - startTime;
    console.log(`Time elapsed: ${timeElapsed}ms, ${synthData.length} points generated`);
  }
})