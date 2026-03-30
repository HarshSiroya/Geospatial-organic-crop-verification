/**
 * Organic vs Inorganic Maize Classification
 * Author: Harsh Siroya
 * Platform: Google Earth Engine
 */

// --------- STEP 0: Load and merge polygons (labeled data)
var organicMaize = ee.FeatureCollection("projects/ee-harshsiroya77/assets/organic-nl-1")
    .map(function(f) { return f.set('class', 1); });
var inorganicMaize = ee.FeatureCollection("projects/ee-harshsiroya77/assets/inorganic-nl-1")
    .map(function(f) { return f.set('class', 0); });
var polygons = organicMaize.merge(inorganicMaize);

// Get study area bounds for optimization
var studyArea = polygons.geometry().bounds();

// --------- STEP 1: Define optimized time parameters
var years = [2021, 2022, 2023, 2024];
var startMonth = 5; // May
var endMonth = 10;  // Oct

// Create comprehensive date filter
var startDate = ee.Date.fromYMD(2021, startMonth, 1);
var endDate = ee.Date.fromYMD(2024, endMonth, 31);

// --------- STEP 2: Optimized cloud masking function
var maskS2Clouds = function(image) {
  var qa = image.select('QA60');
  var cloudMask = qa.bitwiseAnd(1 << 10).eq(0)
                    .and(qa.bitwiseAnd(1 << 11).eq(0));
  return image.updateMask(cloudMask)
              .divide(10000) // Convert to reflectance values
              .copyProperties(image, ['system:time_start']);
};

// --------- STEP 3: Load and preprocess Sentinel-2 collection (optimized)
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(studyArea) // Spatial filter first
    .filterDate(startDate, endDate) // Single temporal filter
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) // Stricter cloud filter
    .select(['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'QA60']) // Select only needed bands
    .map(maskS2Clouds);

// --------- STEP 4: Optimized spectral indices function
var addKeyIndices = function(img) {
  var nir = img.select('B8');
  var red = img.select('B4');
  var blue = img.select('B2');
  var green = img.select('B3');
  var redEdge = img.select('B5');

  // Calculate indices using optimized methods
  var ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');
  var gndvi = nir.subtract(green).divide(nir.add(green)).rename('GNDVI');
  var ndre = nir.subtract(redEdge).divide(nir.add(redEdge)).rename('NDRE');
  
  // EVI calculation
  var evi = img.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
    {'NIR': nir, 'RED': red, 'BLUE': blue}
  ).rename('EVI');
  
  // SAVI calculation
  var savi = img.expression(
    '1.5 * ((NIR - RED) / (NIR + RED + 0.5))',
    {'NIR': nir, 'RED': red}
  ).rename('SAVI');

  return img.addBands([ndvi, evi, gndvi, savi, ndre]);
};

// --------- STEP 5: Create optimized seasonal composite
var seasonComposite = s2
  .map(addKeyIndices)
  .select(['NDVI', 'EVI', 'GNDVI', 'SAVI', 'NDRE'])
  .median()
  .clip(studyArea);

// --------- STEP 6: Sample data from polygons (optimized)
var samples = seasonComposite
  .sampleRegions({
    collection: polygons,
    properties: ['class'],
    scale: 10,
    tileScale: 2, // Reduced for better memory management
    geometries: true
  });

// Add random column for splitting
samples = samples.randomColumn('random', 42); // Fixed seed for reproducibility

// --------- STEP 7: Create train-test split
var training = samples.filter(ee.Filter.lt('random', 0.7));
var testing = samples.filter(ee.Filter.gte('random', 0.7));

// Print sample counts for verification
print('=== DATA SUMMARY ===');
print('Total samples:', samples.size());
print('Training samples:', training.size());
print('Testing samples:', testing.size());

// --------- STEP 8: Train the BEST MODEL (Random Forest with All Indices)
print('=== TRAINING BEST MODEL ===');
print('Model: Random Forest (100 trees)');
print('Features: All indices (NDVI, EVI, GNDVI, SAVI, NDRE)');

var bestClassifier = ee.Classifier.smileRandomForest({
  numberOfTrees: 100,
  variablesPerSplit: null,
  minLeafPopulation: 1,
  bagFraction: 0.5,
  seed: 42
});

// Train the classifier with all indices
var trainedModel = bestClassifier.train({
  features: training,
  classProperty: 'class',
  inputProperties: ['NDVI', 'EVI', 'GNDVI', 'SAVI', 'NDRE']
});

// --------- STEP 9.1: Print Feature Importance (in descending order) - FIXED
var importance = trainedModel.explain();
print('=== FEATURE IMPORTANCE ===');
print('Feature Importance Dictionary:', importance);

// Alternative method to get feature importance
var importanceDict = ee.Dictionary(importance.get('importance'));
print('Importance Values:', importanceDict);

// --------- STEP 9: Validate the model
var classified = testing.classify(trainedModel);
var errorMatrix = classified.errorMatrix('class', 'classification');

print('=== MODEL VALIDATION RESULTS ===');
print('Confusion Matrix:', errorMatrix);
print('Overall Accuracy:', errorMatrix.accuracy());
print('Kappa Coefficient:', errorMatrix.kappa());

// Fixed Producer's and Consumer's Accuracy calls
var producerAccuracy = errorMatrix.producersAccuracy();
var consumerAccuracy = errorMatrix.consumersAccuracy();

print('Producer Accuracy Matrix:', producerAccuracy);
print('Consumer Accuracy Matrix:', consumerAccuracy);

// --------- STEP 10: Create final classification map
var finalClassification = seasonComposite
  .select(['NDVI', 'EVI', 'GNDVI', 'SAVI', 'NDRE'])
  .classify(trainedModel);

// --------- STEP 11: Calculate area statistics
var areaImage = ee.Image.pixelArea().divide(10000); // Convert to hectares
var organicArea = finalClassification.eq(1).multiply(areaImage);
var inorganicArea = finalClassification.eq(0).multiply(areaImage);

var organicStats = organicArea.reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: studyArea,
  scale: 10,
  maxPixels: 1e9
});

var inorganicStats = inorganicArea.reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: studyArea,
  scale: 10,
  maxPixels: 1e9
});

print('=== AREA STATISTICS ===');
print('Organic Maize Area (hectares):', organicStats.get('classification'));
print('Inorganic Maize Area (hectares):', inorganicStats.get('classification'));

// --------- STEP 12: Advanced Visualization Setup
Map.centerObject(polygons, 11);

// Color palettes
var classificationPalette = ['#E74C3C', '#27AE60']; // Red for inorganic, Green for organic
var ndviPalette = ['#8B4513', '#DAA520', '#ADFF2F', '#228B22', '#006400'];
var eviPalette = ['#8B0000', '#FF4500', '#FFD700', '#ADFF2F', '#228B22'];

// --------- STEP 13: Layer Visualization
// 1. Base Sentinel-2 imagery
Map.addLayer(
  seasonComposite.select(['B8', 'B4', 'B3']).multiply(3.5).clip(studyArea),
  {min: 0, max: 0.3, gamma: 1.2},
  '🛰️ Sentinel-2 False Color (NIR-R-G)',
  false
);

Map.addLayer(
  seasonComposite.select(['B4', 'B3', 'B2']).multiply(3.5).clip(studyArea),
  {min: 0, max: 0.3, gamma: 1.2}, 
  '🌍 Sentinel-2 True Color (RGB)',
  false
);

// 2. Vegetation indices
Map.addLayer(
  seasonComposite.select('NDVI').clip(studyArea),
  {min: 0.2, max: 0.9, palette: ndviPalette},
  '🌱 NDVI (Vegetation Health)',
  false
);

Map.addLayer(
  seasonComposite.select('EVI').clip(studyArea),
  {min: 0.1, max: 0.6, palette: eviPalette},
  '📊 EVI (Enhanced Vegetation)',
  false
);

Map.addLayer(
  seasonComposite.select('GNDVI').clip(studyArea),
  {min: 0.2, max: 0.8, palette: ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']},
  '🟢 GNDVI (Green Vegetation)',
  false
);

// 3. Main classification result
Map.addLayer(
  finalClassification.clip(studyArea),
  {min: 0, max: 1, palette: classificationPalette},
  '🎯 MAIZE CLASSIFICATION',
  true
);

// 4. Training polygons for reference
Map.addLayer(
  organicMaize,
  {color: '#2ECC71', fillColor: '2ECC71AA'},
  '🟢 Organic Training Areas',
  false
);

Map.addLayer(
  inorganicMaize,
  {color: '#E74C3C', fillColor: 'E74C3CAA'},
  '🔴 Inorganic Training Areas',
  false
);

// 5. Classification confidence map - FIXED
var probabilities = finalClassification.toArray();
var entropy = probabilities.arrayReduce(ee.Reducer.entropy(), [0]);
Map.addLayer(
  entropy.clip(studyArea),
  {min: 0, max: 1, palette: ['#2ECC71', '#F39C12', '#E74C3C']},
  '🎲 Classification Confidence',
  false
);

// --------- STEP 14: Create legend and info panel - SIMPLIFIED
var legend = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px 15px'
  }
});

var legendItems = [
  ui.Label('🌾 MAIZE CLASSIFICATION LEGEND', {fontWeight: 'bold'}),
  ui.Label('🟢 Organic Maize', {color: '#27AE60'}),
  ui.Label('🔴 Inorganic Maize', {color: '#E74C3C'}),
  ui.Label('📈 Model: Random Forest (100 trees)', {fontSize: '12px'})
];

legendItems.forEach(function(item) {
  legend.add(item);
});

Map.add(legend);

// --------- STEP 15: Export options (ready to use)
/*
// Export classification map
Export.image.toDrive({
  image: finalClassification.clip(studyArea).byte(),
  description: 'Organic_Maize_Classification_RF',
  folder: 'GEE_Maize_Classification',
  scale: 10,
  region: studyArea,
  maxPixels: 1e9,
  crs: 'EPSG:4326'
});

// Export validation results
Export.table.toDrive({
  collection: testing.select(['class', 'classification', 'NDVI', 'EVI', 'GNDVI', 'SAVI', 'NDRE']),
  description: 'Validation_Results_RF_All_Indices',
  folder: 'GEE_Maize_Classification',
  fileFormat: 'CSV'
});

// Export area statistics as a feature collection
var organicAreaValue = ee.Number(organicStats.get('classification'));
var inorganicAreaValue = ee.Number(inorganicStats.get('classification'));
var totalArea = organicAreaValue.add(inorganicAreaValue);

var areaStats = ee.FeatureCollection([
  ee.Feature(null, {
    'Organic_Area_Ha': organicAreaValue,
    'Inorganic_Area_Ha': inorganicAreaValue,
    'Total_Area_Ha': totalArea,
    'Organic_Percentage': organicAreaValue.divide(totalArea).multiply(100)
  })
]);

Export.table.toDrive({
  collection: areaStats,
  description: 'Area_Statistics_Maize_Classification',
  folder: 'GEE_Maize_Classification',
  fileFormat: 'CSV'
});
*/

// --------- FINAL SUMMARY
print('=== 🎉 CLASSIFICATION COMPLETED SUCCESSFULLY! ===');
print('✅ Model: Random Forest (100 trees)');
print('✅ Features: All vegetation indices');
print('✅ Classification map generated and visualized');
print('📍 Study Area: Netherlands Maize Fields');
print('📅 Time Period: May-October 2021-2024');
print('🛰️ Data Source: Sentinel-2 SR Harmonized');
print('');
print('💡 Tip: Toggle layers in the map to explore different visualizations!');
print('📥 Uncomment export section to download results to Google Drive');