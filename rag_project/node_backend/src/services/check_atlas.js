const path = require('path');
require('dotenv').config({ path: path.resolve(__dirname, '../../.env') });

const { MongoClient } = require('mongodb');
const { embedTexts } = require('./embedding');

async function testVectorIndex() {
  const client = new MongoClient(process.env.MONGODB_ATLAS_URI);
  
  try {
    await client.connect();
    console.log('✅ Connected to MongoDB\n');
    
    const db = client.db('document_ai');
    const collection = db.collection('DocumentChunks');
    
    // Step 1: Check documents exist
    const docCount = await collection.countDocuments();
    console.log(`Total documents: ${docCount}`);
    
    if (docCount === 0) {
      console.log('❌ No documents found! Upload documents first.');
      return;
    }
    
    // Step 2: Generate test query embedding
    console.log('\nGenerating test query embedding...');
    const [queryVector] = await embedTexts(['what is average']);
    console.log(`✅ Query vector: ${queryVector.length} dimensions`);
    
    // Step 3: Test vector search
    console.log('\nTesting vector search...');
    
    const pipeline = [
      {
        $vectorSearch: {
          index: 'vector_index',
          path: 'embedding',
          queryVector: queryVector,
          numCandidates: 50,
          limit: 3
        }
      },
      {
        $project: {
          text: 1,
          documentId: 1,
          score: { $meta: 'vectorSearchScore' }
        }
      }
    ];
    
    const results = await collection.aggregate(pipeline).toArray();
    
    if (results.length > 0) {
      console.log(`\n✅ SUCCESS! Vector search is working!`);
      console.log(`Found ${results.length} results:\n`);
      
      results.forEach((result, i) => {
        console.log(`Result ${i + 1}:`);
        console.log(`  Score: ${result.score.toFixed(4)}`);
        console.log(`  Document: ${result.documentId.substring(0, 50)}...`);
        console.log(`  Text: ${result.text.substring(0, 100)}...`);
        console.log();
      });
      
      console.log('🎉 Your vector search is ready to use!');
      
    } else {
      console.log('\n❌ No results found!');
      console.log('Possible issues:');
      console.log('1. Vector index not active yet (wait 2-5 minutes)');
      console.log('2. Index configuration incorrect');
      console.log('3. Check index name is "vector_index"');
      console.log('4. Check numDimensions is 384');
    }
    
  } catch (error) {
    console.error('\n❌ Error:', error.message);
    
    if (error.message.includes('$vectorSearch')) {
      console.log('\n🔧 FIX: Vector search index not found or not active');
      console.log('1. Go to Atlas → Atlas Search');
      console.log('2. Check index status is ACTIVE/READY');
      console.log('3. Wait a few minutes if status is BUILDING');
      console.log('4. Recreate index if status is FAILED');
    }
    
  } finally {
    await client.close();
  }
}

testVectorIndex();
