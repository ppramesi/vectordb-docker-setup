import Knex, { Knex as KnexT } from "knex";
import * as dotenv from "dotenv";
import { generateSyntheticData, generateSyntheticDataFast } from "../utils/math";

dotenv.config();

const dimensions = 1000;
const pointNums = 100;

const max = 10;
const clusterNum = 100;

const pgvectorKnexConfig: KnexT.Config = {
  client: "postgresql",
  connection: {
    host: process.env.POSTGRES_HOST,
    database: process.env.POSTGRES_PGVECTOR_DB,
    user: process.env.POSTGRES_PGVECTOR_USER,
    password: process.env.POSTGRES_PGVECTOR_PASSWORD,
    port: Number(process.env.POSTGRES_PGVECTOR_PORT)
  },
  pool: { min: 2, max: 20 }
}

const pgembeddingKnexConfig: KnexT.Config = {
  client: "postgresql",
  connection: {
    host: process.env.POSTGRES_HOST,
    database: process.env.POSTGRES_PGEMBEDDING_DB,
    user: process.env.POSTGRES_PGEMBEDDING_USER,
    password: process.env.POSTGRES_PGEMBEDDING_PASSWORD,
    port: Number(process.env.POSTGRES_PGEMBEDDING_PORT)
  },
  pool: { min: 2, max: 20 }
}

const letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"];

async function insertVectors(knex: KnexT, tableName: string, chunks: { embedding: number[], category: string }[], ext: "pgvector" | "pgembedding"){
  const values = chunks.map((chunk) => {
    const arrayValue = ext === "pgvector" ? 
      `'[${chunk.embedding.join(",")}]'` : 
      `'{${chunk.embedding.join(",")}}'`;
    return `(${arrayValue}, '${chunk.category}')`;
  }).join(", ");

  return knex.raw(`INSERT INTO ${tableName} (embedding, category) VALUES ${values};`);
}

async function up(knex: KnexT, ext: "pgvector" | "pgembedding"){
  const analyzeTable = (tableName: string) => {
    return knex.schema.raw(`ANALYZE ${tableName};`);
  }
  const buildIndexQuery = (idxName: string, tableName: string, semicolon: boolean = true) => {
    if(ext === "pgvector"){
      return `CREATE INDEX IF NOT EXISTS ${idxName} ON ${tableName} USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)${semicolon ? ";" : ""}`;
    }else if(ext === "pgembedding"){
      return `CREATE INDEX IF NOT EXISTS ${idxName} ON ${tableName} USING hnsw (embedding ann_cos_ops) WITH (dims = ${dimensions}, m = 16, efconstruction = 64, efsearch = 40)${semicolon ? ";" : ""}`;
    }else{
      throw new Error("Unsupported extension");
    }
  }

  const buildPartialIndexQuery = (idxName: string, tableName: string, letter: string) => {
    return `${buildIndexQuery(idxName, tableName, false)} WHERE category = '${letter}';`;
  }
  
  await knex.schema.raw(`CREATE TABLE IF NOT EXISTS vectors_partitioned (embedding ${ext === "pgvector" ? `vector(${dimensions})` : "real[]"}, category text) PARTITION BY LIST (category);`);
  console.log("Creating partitions")
  await Promise.all(letters.map((letter) => {
    return knex.schema.raw(`CREATE TABLE IF NOT EXISTS vectors_partitioned_${letter} PARTITION OF vectors_partitioned FOR VALUES IN ('${letter}');`);
  }))
  console.log("Creating indexes")
  await Promise.all(letters.map((letter) => {
    return knex.schema.raw(buildIndexQuery(`vectors_partitioned_${letter}_index`, `vectors_partitioned_${letter}`, true));
  }));
  await analyzeTable("vectors_partitioned");
  console.log("Done creating indexes")

  await knex.schema.createTableIfNotExists("vectors_full_index", (table) => {
    table.uuid("id").primary().defaultTo(knex.raw("uuid_generate_v4()"));
    table.specificType("embedding", ext === "pgvector" ? `vector(${dimensions})` : "real[]");
    table.enum("category", letters);
  });
  console.log("Creating full index")

  await knex.schema.raw(buildIndexQuery("vectors_full_index_index", "vectors_full_index"));
  await analyzeTable("vectors_full_index");
  console.log("Done creating full index")

  await knex.schema.createTableIfNotExists("vectors_partial_index", (table) => {
    table.uuid("id").primary().defaultTo(knex.raw("uuid_generate_v4()"));
    table.specificType("embedding", ext === "pgvector" ? `vector(${dimensions})` : "real[]");
    table.enum("category", letters);
  });
  await analyzeTable("vectors_partial_index");
  console.log("Creating partial index")

  await Promise.all(letters.map((letter) => {
    return knex.schema.raw(buildPartialIndexQuery(`vectors_partial_index_${letter}_index`, "vectors_partial_index", letter));
  }));
  console.log("Done creating partial index")

  await knex.schema.createTableIfNotExists("vectors_no_index", (table) => {
    table.uuid("id").primary().defaultTo(knex.raw("uuid_generate_v4()"));
    table.specificType("embedding", ext === "pgvector" ? `vector(${dimensions})` : "real[]");
    table.enum("category", letters);
  });
  await analyzeTable("vectors_no_index");
  console.log("Done creating no index")
}

async function down(knex: KnexT){
  await Promise.all(letters.map((letter) => {
    return knex.schema.dropTableIfExists(`vectors_partitioned_${letter}`);
  })).catch(console.error);

  await knex.schema.raw("DROP INDEX IF EXISTS vectors_full_index_index;");
  await Promise.all(letters.map((letter) => {
    return knex.schema.raw(`DROP INDEX IF EXISTS vectors_partial_index_${letter}_index;`);
  })).catch(console.error);

  await Promise.all(letters.map((letter) => {
    return knex.schema.raw(`DROP INDEX IF EXISTS vectors_partitioned_${letter}_index;`);
  })).catch(console.error);

  await Promise.all([
    knex.schema.dropTableIfExists("vectors_partial_index").catch(console.error),
    knex.schema.dropTableIfExists("vectors_no_index").catch(console.error),
    knex.schema.dropTableIfExists("vectors_full_index").catch(console.error),
    knex.schema.dropTableIfExists("vectors_partitioned").catch(console.error)
  ])
}

let pgvectorKnex: KnexT;
let pgembeddingKnex: KnexT;

beforeAll(async () => {
  pgvectorKnex = Knex(pgvectorKnexConfig);
  pgembeddingKnex = Knex(pgembeddingKnexConfig);

  console.log("Dropping tables")
  await Promise.all([
    down(pgvectorKnex),
    down(pgembeddingKnex)
  ])
  console.log("Done dropping tables")

  try {
    await up(pgvectorKnex, "pgvector");
    await up(pgembeddingKnex, "pgembedding");
  } catch (error) {
    await Promise.all([
      down(pgvectorKnex),
      down(pgembeddingKnex)
    ])
  
    await Promise.all([
      pgvectorKnex.destroy(),
      pgembeddingKnex.destroy()
    ])
  }

  for(let i = 0; i < max; i++){
    const startTime = performance.now();
    const synthData = generateSyntheticDataFast(clusterNum, pointNums, dimensions); // 100 clusters, pointNums points per cluster, 10 dimensions
    const synthDataWithCategories = synthData.map((vector) => {
      return {
        embedding: vector,
        category: letters[Math.floor(Math.random() * letters.length)]
      }
    });

    const chunks = [];
    for(let i = 0; i < Math.floor((clusterNum * pointNums) / 1000); i++){
      chunks.push(synthDataWithCategories.slice(i * 1000, (i + 1) * 1000));
    }
    console.log(`Done generating synthetic data. Time elapsed: ${performance.now() - startTime}ms`);
    // we insert the chunks into the database
    const data: Record<string, Record<string, number[]>> = {
      pgembedding: {
        vectors_partitioned: [],
        vectors_full_index: [],
        vectors_partial_index: [],
        vectors_no_index: []
      },
      pgvector: {
        vectors_partitioned: [],
        vectors_full_index: [],
        vectors_partial_index: [],
        vectors_no_index: []
      }
    };

    const insertAndMeasure = (knexInstance: KnexT, idxType: string, chunk: any, extType: "pgvector" | "pgembedding") => {
      const innerStartTime = performance.now();
      return insertVectors(knexInstance, idxType, chunk, extType).then(() => {
        const finishedTime = performance.now() - innerStartTime;
        data[extType][idxType].push(finishedTime);
      })
    }

    for(const chunk of chunks){
      await insertAndMeasure(pgembeddingKnex, "vectors_no_index", chunk, "pgembedding");
      await insertAndMeasure(pgembeddingKnex, "vectors_partitioned", chunk, "pgembedding");
      await insertAndMeasure(pgembeddingKnex, "vectors_full_index", chunk, "pgembedding");
      await insertAndMeasure(pgembeddingKnex, "vectors_partial_index", chunk, "pgembedding");

      await insertAndMeasure(pgvectorKnex, "vectors_no_index", chunk, "pgvector");
      await insertAndMeasure(pgvectorKnex, "vectors_partitioned", chunk, "pgvector");
      await insertAndMeasure(pgvectorKnex, "vectors_full_index", chunk, "pgvector");
      await insertAndMeasure(pgvectorKnex, "vectors_partial_index", chunk, "pgvector");
    }

    // create new object that unroll data object and average the arrays
    const averagedData: Record<string, Record<string, number>> = {
      pgembedding: {
        vectors_partitioned: data.pgembedding.vectors_partitioned.reduce((a, b) => a + b, 0) / data.pgembedding.vectors_partitioned.length,
        vectors_full_index: data.pgembedding.vectors_full_index.reduce((a, b) => a + b, 0) / data.pgembedding.vectors_full_index.length,
        vectors_partial_index: data.pgembedding.vectors_partial_index.reduce((a, b) => a + b, 0) / data.pgembedding.vectors_partial_index.length,
        vectors_no_index: data.pgembedding.vectors_no_index.reduce((a, b) => a + b, 0) / data.pgembedding.vectors_no_index.length
      },
      pgvector: {
        vectors_partitioned: data.pgvector.vectors_partitioned.reduce((a, b) => a + b, 0) / data.pgvector.vectors_partitioned.length,
        vectors_full_index: data.pgvector.vectors_full_index.reduce((a, b) => a + b, 0) / data.pgvector.vectors_full_index.length,
        vectors_partial_index: data.pgvector.vectors_partial_index.reduce((a, b) => a + b, 0) / data.pgvector.vectors_partial_index.length,
        vectors_no_index: data.pgvector.vectors_no_index.reduce((a, b) => a + b, 0) / data.pgvector.vectors_no_index.length
      }
    }
    console.table(averagedData);
  }
})

function doQuery(ext: "pgvector" | "pgembedding", myKnex: KnexT, query: (db: KnexT) => KnexT.QueryBuilder | KnexT.Raw){
  if(ext === "pgvector"){
    return query(myKnex)
  }else{
    return myKnex.transaction((trx) => {
      return trx
        .raw("SET LOCAL enable_seqscan = off;")
        .then(() => {
          return query(trx);
        })
    })
  }
}

async function executeExplainAnalyze(knex: KnexT, tableName: string, queryVector: number[], letter: string, ext: "pgvector" | "pgembedding", print: boolean = false): Promise<void> {
  let arrayValue: any = ext === "pgvector" ? `[${queryVector.join(",")}]` : knex.raw(`array[${queryVector.join(",")}]`);
  const result = await doQuery(ext, knex, (db) => db.raw(`EXPLAIN ANALYZE SELECT * FROM ${tableName} WHERE category = ? ORDER BY embedding <=> ? LIMIT 100`, [letter, arrayValue]));
  // const result = await knex.raw(`EXPLAIN ANALYZE SELECT * FROM ${tableName} WHERE category = ? ORDER BY embedding <=> ? LIMIT 100`, [letter, arrayValue]);
  if(print){
    console.log(`EXPLAIN ANALYZE for table ${tableName} with ${ext}:\n`, result.rows);
  }
}

async function executeExplainAnalyzeNoWhere(knex: KnexT, tableName: string, queryVector: number[], ext: "pgvector" | "pgembedding", print: boolean = false): Promise<void> {
  let arrayValue: any = ext === "pgvector" ? `[${queryVector.join(",")}]` : knex.raw(`array[${queryVector.join(",")}]`);
  const result = await doQuery(ext, knex, (db) => db.raw(`EXPLAIN ANALYZE SELECT * FROM ${tableName} ORDER BY embedding <=> ? LIMIT 100`, [arrayValue]));
  // const result = await knex.raw(`EXPLAIN ANALYZE SELECT * FROM ${tableName} ORDER BY embedding <=> ? LIMIT 100`, [arrayValue]);
  if(print){
    console.log(`EXPLAIN ANALYZE for table ${tableName} with ${ext} nowhere:\n`, result.rows);
  }
}

async function executeSelect(knex: KnexT, tableName: string, queryVector: number[], letter: string, ext: "pgvector" | "pgembedding", print: boolean = false): Promise<void> {
  let arrayValue: any = ext === "pgvector" ? `[${queryVector.join(",")}]` : knex.raw(`array[${queryVector.join(",")}]`);
  const result = await doQuery(ext, knex, (db) => db.raw(`SELECT * FROM ${tableName} WHERE category = ? ORDER BY embedding <=> ? LIMIT 100`, [letter, arrayValue]));
  // const result = await knex.raw(`SELECT * FROM ${tableName} WHERE category = ? ORDER BY embedding <=> ? LIMIT 100`, [letter, arrayValue]);
  if(print){
    console.log(`SELECT for table ${tableName} with ${ext}:\n`, result.rows);
  }

  return result;
}

async function executeSelectNoWhere(knex: KnexT, tableName: string, queryVector: number[], ext: "pgvector" | "pgembedding", print: boolean = false): Promise<void> {
  let arrayValue: any = ext === "pgvector" ? `[${queryVector.join(",")}]` : knex.raw(`array[${queryVector.join(",")}]`);
  const result = await doQuery(ext, knex, (db) => db.raw(`SELECT * FROM ${tableName} ORDER BY embedding <=> ? LIMIT 100`, [arrayValue]));
  // const result = await knex.raw(`SELECT * FROM ${tableName} ORDER BY embedding <=> ? LIMIT 100`, [arrayValue]);
  if(print){
    console.log(`SELECT for table ${tableName} with ${ext} nowhere:\n`, result.rows);
  }

  return result;
}

const warmupIterations = 50;

test("Main test", async () => {
  const results = {
    pgvector: {
      vectors_partitioned: [],
      vectors_full_index: [],
      vectors_partial_index: [],
      vectors_no_index: []
    },
    pgembedding: {
      vectors_partitioned: [],
      vectors_full_index: [],
      vectors_partial_index: [],
      vectors_no_index: []
    }
  };
  const resultsNowhere = {
    pgvector: {
      vectors_partitioned: [],
      vectors_full_index: [],
      vectors_partial_index: [],
      vectors_no_index: []
    },
    pgembedding: {
      vectors_partitioned: [],
      vectors_full_index: [],
      vectors_partial_index: [],
      vectors_no_index: []
    }
  };
  const doStuffNowhere = async (pgInstance: KnexT, idxType: string, extType: "pgvector" | "pgembedding") => {
    const startTime = performance.now();
    console.log("doing stuff nowhere, warming up...");
    for (let i = 0; i < warmupIterations; i++) {
      const queryVector = generateSyntheticData(1, 1, dimensions)[0];
      await executeSelectNoWhere(pgInstance, idxType, queryVector, extType, false);
    }
    console.log("done warming up, actually running now...");
  
    for(let i = 0; i < 100; i++){
      const queryVector = generateSyntheticData(1, 1, dimensions)[0];
      const startTime = performance.now();
      await executeSelectNoWhere(pgInstance, idxType, queryVector, extType, false);
      const elapsedTime = performance.now() - startTime;
      resultsNowhere[extType][idxType].push(elapsedTime);
    }
    console.log(`done running, elapsed time: ${performance.now() - startTime}ms`);
  }
  const doStuff = async (pgInstance: KnexT, idxType: string, extType: "pgvector" | "pgembedding") => {
    const startTime = performance.now();
    console.log("doing stuff, warming up...");
    for (let i = 0; i < warmupIterations; i++) {
      const queryVector = generateSyntheticData(1, 1, dimensions)[0];
      const randomLetter = letters[Math.floor(Math.random() * letters.length)];
      await executeSelect(pgInstance, idxType, queryVector, randomLetter, extType, false);
    }
    console.log("done warming up, actually running now...");
  
    for(let i = 0; i < 100; i++){
      const queryVector = generateSyntheticData(1, 1, dimensions)[0];
      const randomLetter = letters[Math.floor(Math.random() * letters.length)];
      const startTime = performance.now();
      await executeSelect(pgInstance, idxType, queryVector, randomLetter, extType, false);
      const elapsedTime = performance.now() - startTime;
      results[extType][idxType].push(elapsedTime);
    }
    console.log(`done running, elapsed time: ${performance.now() - startTime}ms`);
  }

  await doStuff(pgvectorKnex, "vectors_partitioned", "pgvector");
  await doStuff(pgvectorKnex, "vectors_full_index", "pgvector");
  await doStuff(pgvectorKnex, "vectors_partial_index", "pgvector");
  await doStuff(pgvectorKnex, "vectors_no_index", "pgvector");

  await doStuff(pgembeddingKnex, "vectors_partitioned", "pgembedding");
  await doStuff(pgembeddingKnex, "vectors_full_index", "pgembedding");
  await doStuff(pgembeddingKnex, "vectors_partial_index", "pgembedding");
  await doStuff(pgembeddingKnex, "vectors_no_index", "pgembedding");

  await doStuffNowhere(pgvectorKnex, "vectors_partitioned", "pgvector");
  await doStuffNowhere(pgvectorKnex, "vectors_full_index", "pgvector");
  await doStuffNowhere(pgvectorKnex, "vectors_partial_index", "pgvector");
  await doStuffNowhere(pgvectorKnex, "vectors_no_index", "pgvector");

  await doStuffNowhere(pgembeddingKnex, "vectors_partitioned", "pgembedding");
  await doStuffNowhere(pgembeddingKnex, "vectors_full_index", "pgembedding");
  await doStuffNowhere(pgembeddingKnex, "vectors_partial_index", "pgembedding");
  await doStuffNowhere(pgembeddingKnex, "vectors_no_index", "pgembedding");

  // average the results
  const averagedResults = {
    pgvector: {
      vectors_partitioned: results.pgvector.vectors_partitioned.reduce((a, b) => a + b, 0) / results.pgvector.vectors_partitioned.length,
      vectors_full_index: results.pgvector.vectors_full_index.reduce((a, b) => a + b, 0) / results.pgvector.vectors_full_index.length,
      vectors_partial_index: results.pgvector.vectors_partial_index.reduce((a, b) => a + b, 0) / results.pgvector.vectors_partial_index.length,
      vectors_no_index: results.pgvector.vectors_no_index.reduce((a, b) => a + b, 0) / results.pgvector.vectors_no_index.length
    },
    pgembedding: {
      vectors_partitioned: results.pgembedding.vectors_partitioned.reduce((a, b) => a + b, 0) / results.pgembedding.vectors_partitioned.length,
      vectors_full_index: results.pgembedding.vectors_full_index.reduce((a, b) => a + b, 0) / results.pgembedding.vectors_full_index.length,
      vectors_partial_index: results.pgembedding.vectors_partial_index.reduce((a, b) => a + b, 0) / results.pgembedding.vectors_partial_index.length,
      vectors_no_index: results.pgembedding.vectors_no_index.reduce((a, b) => a + b, 0) / results.pgembedding.vectors_no_index.length
    }
  }
  const averagedResultsNowhere = {
    pgvector: {
      vectors_partitioned: resultsNowhere.pgvector.vectors_partitioned.reduce((a, b) => a + b, 0) / resultsNowhere.pgvector.vectors_partitioned.length,
      vectors_full_index: resultsNowhere.pgvector.vectors_full_index.reduce((a, b) => a + b, 0) / resultsNowhere.pgvector.vectors_full_index.length,
      vectors_partial_index: resultsNowhere.pgvector.vectors_partial_index.reduce((a, b) => a + b, 0) / resultsNowhere.pgvector.vectors_partial_index.length,
      vectors_no_index: resultsNowhere.pgvector.vectors_no_index.reduce((a, b) => a + b, 0) / resultsNowhere.pgvector.vectors_no_index.length
    },
    pgembedding: {
      vectors_partitioned: resultsNowhere.pgembedding.vectors_partitioned.reduce((a, b) => a + b, 0) / resultsNowhere.pgembedding.vectors_partitioned.length,
      vectors_full_index: resultsNowhere.pgembedding.vectors_full_index.reduce((a, b) => a + b, 0) / resultsNowhere.pgembedding.vectors_full_index.length,
      vectors_partial_index: resultsNowhere.pgembedding.vectors_partial_index.reduce((a, b) => a + b, 0) / resultsNowhere.pgembedding.vectors_partial_index.length,
      vectors_no_index: resultsNowhere.pgembedding.vectors_no_index.reduce((a, b) => a + b, 0) / resultsNowhere.pgembedding.vectors_no_index.length
    }
  }
  console.table(averagedResults);
  console.table(averagedResultsNowhere);
});

afterAll(async () => {
  console.log("Destroying knex connections")
  await Promise.all([
    pgvectorKnex.destroy(),
    pgembeddingKnex.destroy()
  ])
  console.log("Done destroying knex connections")
})

// test("", async () => {
//   const queryVector = generateSyntheticData(1, 1, dimensions)[0];
//   const randomLetter = letters[Math.floor(Math.random() * letters.length)];
//   const pgvResult = await executeSelect(pgvectorKnex, "vectors_full_index", queryVector, randomLetter, "pgvector", true);
//   const pgeResult = await executeSelect(pgembeddingKnex, "vectors_full_index", queryVector, randomLetter, "pgembedding", true);

//   console.log({pgvResult, pgeResult});
// })

// test("Can we actually get data from pgvectors?", async () => {
//   const result = await pgvectorKnex.raw(`SELECT * FROM vectors_no_index LIMIT 10`);
//   expect(result.rows.length).toBeGreaterThan(0);
// })
// test("Can we actually get data from pgembedding?", async () => {
//   const result = await pgembeddingKnex.raw(`SELECT * FROM vectors_no_index LIMIT 10`);
//   expect(result.rows.length).toBeGreaterThan(0);
// })

// test("Checking indexes pgvector", async () => {
//   const result = await pgvectorKnex.raw(`SELECT * FROM pg_indexes WHERE tablename = 'vectors_full_index'`);
//   console.log(result.rows);
// })
// test("Checking indexes pgembedding", async () => {
//   const result = await pgembeddingKnex.raw(`SELECT * FROM pg_indexes WHERE tablename = 'vectors_full_index'`);
//   console.log(result.rows);
// })

// test("Performance tests pgvector partitioned", async () => {
//   for (let i = 0; i < warmupIterations; i++) {
//     const queryVector = generateSyntheticData(1, 1, dimensions)[0];
//     const randomLetter = letters[Math.floor(Math.random() * letters.length)];
//     await executeExplainAnalyze(pgvectorKnex, "vectors_partitioned", queryVector, randomLetter, "pgvector", false);
//   }

//   const queryVector = generateSyntheticData(1, 1, dimensions)[0];
//   const randomLetter = letters[Math.floor(Math.random() * letters.length)];
//   await executeExplainAnalyze(pgvectorKnex, "vectors_partitioned", queryVector, randomLetter, "pgvector", true);
// })
// test("Performance tests pgvector partial index", async () => {
//   for (let i = 0; i < warmupIterations; i++) {
//     const queryVector = generateSyntheticData(1, 1, dimensions)[0];
//     const randomLetter = letters[Math.floor(Math.random() * letters.length)];
//     await executeExplainAnalyze(pgvectorKnex, "vectors_partial_index", queryVector, randomLetter, "pgvector", false);
//   }

//   const queryVector = generateSyntheticData(1, 1, dimensions)[0];
//   const randomLetter = letters[Math.floor(Math.random() * letters.length)];
//   await executeExplainAnalyze(pgvectorKnex, "vectors_partial_index", queryVector, randomLetter, "pgvector", true);
// })
// test("Performance tests pgvector full index", async () => {
//   for (let i = 0; i < warmupIterations; i++) {
//     const queryVector = generateSyntheticData(1, 1, dimensions)[0];
//     const randomLetter = letters[Math.floor(Math.random() * letters.length)];
//     await executeExplainAnalyze(pgvectorKnex, "vectors_full_index", queryVector, randomLetter, "pgvector", false);
//   }

//   const queryVector = generateSyntheticData(1, 1, dimensions)[0];
//   const randomLetter = letters[Math.floor(Math.random() * letters.length)];
//   await executeExplainAnalyze(pgvectorKnex, "vectors_full_index", queryVector, randomLetter, "pgvector", true);
// })
// test("Performance tests pgvector no index", async () => {
//   for (let i = 0; i < warmupIterations; i++) {
//     const queryVector = generateSyntheticData(1, 1, dimensions)[0];
//     const randomLetter = letters[Math.floor(Math.random() * letters.length)];
//     await executeExplainAnalyze(pgvectorKnex, "vectors_no_index", queryVector, randomLetter, "pgvector", false);
//   }

//   const queryVector = generateSyntheticData(1, 1, dimensions)[0];
//   const randomLetter = letters[Math.floor(Math.random() * letters.length)];
//   await executeExplainAnalyze(pgvectorKnex, "vectors_no_index", queryVector, randomLetter, "pgvector", true);
// })
// test("Performance tests pgvector no where full index", async () => {
//   for (let i = 0; i < warmupIterations; i++) {
//     const queryVector = generateSyntheticData(1, 1, dimensions)[0];
//     const randomLetter = letters[Math.floor(Math.random() * letters.length)];
//     await executeExplainAnalyzeNoWhere(pgvectorKnex, "vectors_full_index", queryVector, "pgvector", false);
//   }

//   const queryVector = generateSyntheticData(1, 1, dimensions)[0];
//   const randomLetter = letters[Math.floor(Math.random() * letters.length)];
//   await executeExplainAnalyzeNoWhere(pgvectorKnex, "vectors_full_index", queryVector, "pgvector", true);
// })

// test("Performance tests pgembedding partitioned", async () => {
//   for (let i = 0; i < warmupIterations; i++) {
//     const queryVector = generateSyntheticData(1, 1, dimensions)[0];
//     const randomLetter = letters[Math.floor(Math.random() * letters.length)];
//     await executeExplainAnalyze(pgembeddingKnex, "vectors_partitioned", queryVector, randomLetter, "pgembedding", false);
//   }

//   const queryVector = generateSyntheticData(1, 1, dimensions)[0];
//   const randomLetter = letters[Math.floor(Math.random() * letters.length)];
//   await executeExplainAnalyze(pgembeddingKnex, "vectors_partitioned", queryVector, randomLetter, "pgembedding", true);
// })
// test("Performance tests pgembedding partial index", async () => {
//   for (let i = 0; i < warmupIterations; i++) {
//     const queryVector = generateSyntheticData(1, 1, dimensions)[0];
//     const randomLetter = letters[Math.floor(Math.random() * letters.length)];
//     await executeExplainAnalyze(pgembeddingKnex, "vectors_partial_index", queryVector, randomLetter, "pgembedding", false);
//   }

//   const queryVector = generateSyntheticData(1, 1, dimensions)[0];
//   const randomLetter = letters[Math.floor(Math.random() * letters.length)];
//   await executeExplainAnalyze(pgembeddingKnex, "vectors_partial_index", queryVector, randomLetter, "pgembedding", true);
// })
// test("Performance tests pgembedding full index", async () => {
//   for (let i = 0; i < warmupIterations; i++) {
//     const queryVector = generateSyntheticData(1, 1, dimensions)[0];
//     const randomLetter = letters[Math.floor(Math.random() * letters.length)];
//     await executeExplainAnalyze(pgembeddingKnex, "vectors_full_index", queryVector, randomLetter, "pgembedding", false);
//   }

//   const queryVector = generateSyntheticData(1, 1, dimensions)[0];
//   const randomLetter = letters[Math.floor(Math.random() * letters.length)];
//   await executeExplainAnalyze(pgembeddingKnex, "vectors_full_index", queryVector, randomLetter, "pgembedding", true);
// })
// test("Performance tests pgembedding no index", async () => {
//   for (let i = 0; i < warmupIterations; i++) {
//     const queryVector = generateSyntheticData(1, 1, dimensions)[0];
//     const randomLetter = letters[Math.floor(Math.random() * letters.length)];
//     await executeExplainAnalyze(pgembeddingKnex, "vectors_no_index", queryVector, randomLetter, "pgembedding", false);
//   }

//   const queryVector = generateSyntheticData(1, 1, dimensions)[0];
//   const randomLetter = letters[Math.floor(Math.random() * letters.length)];
//   await executeExplainAnalyze(pgembeddingKnex, "vectors_no_index", queryVector, randomLetter, "pgembedding", true);
// })
// test("Performance tests pgembedding no where full index", async () => {
//   for (let i = 0; i < warmupIterations; i++) {
//     const queryVector = generateSyntheticData(1, 1, dimensions)[0];
//     await executeExplainAnalyzeNoWhere(pgembeddingKnex, "vectors_full_index", queryVector, "pgembedding", false);
//   }
  
//   const queryVector = generateSyntheticData(1, 1, dimensions)[0];
//   await executeExplainAnalyzeNoWhere(pgembeddingKnex, "vectors_full_index", queryVector, "pgembedding", true);
// })