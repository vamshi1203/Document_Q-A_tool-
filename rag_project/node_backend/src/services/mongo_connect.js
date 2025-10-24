// mongo_connect.js
const path = require('path');
require('dotenv').config({ path: path.resolve(__dirname, '../../.env') });
const { MongoClient } = require('mongodb');

console.log('MONGODB_ATLAS_URI:', process.env.MONGODB_ATLAS_URI);
async function testConnection() {
  console.log('🔍 Testing MongoDB Atlas Connection...\n');
  
  const uri = process.env.MONGODB_ATLAS_URI;
  
  if (!uri) {
    console.error('❌ MONGODB_ATLAS_URI not found in .env file!');
    process.exit(1);
  }
  
  const safeUri = uri.replace(/:[^:@]+@/, ':****@');
  console.log('Connection URI:', safeUri);
  console.log('');
  
  const client = new MongoClient(uri);
  
  try {
    console.log('Connecting to MongoDB Atlas...');
    await client.connect();
    console.log('✅ Connection successful!\n');
    
    // Test 1: Ping database
    console.log('Test 1: Pinging database...');
    const admin = client.db('admin');
    const pingResult = await admin.command({ ping: 1 });
    console.log('✅ Ping successful:', pingResult);
    console.log('');
    
    // Test 2: List databases
    console.log('Test 2: Listing databases...');
    const dbList = await admin.admin().listDatabases();
    console.log('✅ Databases:');
    dbList.databases.forEach(db => {
      console.log(`   - ${db.name} (${(db.sizeOnDisk / 1024 / 1024).toFixed(2)} MB)`);
    });
    console.log('');
    
    // Test 3: Check document_ai database
    console.log('Test 3: Checking document_ai database...');
    const db = client.db('document_ai');
    const collections = await db.listCollections().toArray();
    console.log(`✅ Collections in document_ai: ${collections.length}`);
    collections.forEach(col => {
      console.log(`   - ${col.name}`);
    });
    console.log('');
    
    // Test 4: Check DocumentChunks collection
    if (collections.some(c => c.name === 'DocumentChunks')) {
      console.log('Test 4: Checking DocumentChunks collection...');
      const collection = db.collection('DocumentChunks');
      const docCount = await collection.countDocuments();
      console.log(`✅ DocumentChunks has ${docCount} documents`);
      
      if (docCount > 0) {
        const sample = await collection.findOne();
        console.log('Sample document structure:');
        console.log('   - Fields:', Object.keys(sample));
        console.log('   - Has embedding:', !!sample.embedding);
        console.log('   - Embedding dimensions:', sample.embedding?.length || 0);
      } else {
        console.log('⚠️  Collection is empty - this is why vector index failed!');
      }
    } else {
      console.log('⚠️  DocumentChunks collection does not exist yet');
    }
    
    console.log('\n✅ All tests passed! MongoDB connection is working.');
    console.log('\n💡 Next steps:');
    console.log('1. Upload a document via /api/ingest to add data');
    console.log('2. Check that documents have "embedding" field with 384 dimensions');
    console.log('3. Delete and recreate the vector search index in Atlas UI');
    
  } catch (error) {
    console.error('\n❌ Connection failed!');
    console.error('Error:', error.message);
    
    if (error.message.includes('bad auth')) {
      console.error('\n🔧 Fix: Wrong username or password');
      console.error('1. Go to MongoDB Atlas → Database Access');
      console.error('2. Verify username: raguser');
      console.error('3. Reset password if needed');
      console.error('4. Update .env with correct password');
    } else if (error.message.includes('ENOTFOUND')) {
      console.error('\n🔧 Fix: Check cluster URL');
    } else if (error.message.includes('timeout')) {
      console.error('\n🔧 Fix: Check Network Access in Atlas');
      console.error('Add your IP: Atlas → Network Access → Add IP Address');
    }
    
  } finally {
    await client.close();
    console.log('\n🔌 Connection closed');
  }
}

testConnection()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
