#!/bin/bash
# EmTechScan – Custom Arial Model Trainer (Loop-Safe Version)

DATASET_DIR="./tesseract_train_data"
ENGINE="./engine"

echo "🧠 Generating .tr feature files..."
for f in "$DATASET_DIR"/*.tif; do
  $ENGINE/tesseract "$f" "${f%.tif}" -l eng nobatch box.train
done
echo "✅ .tr feature generation complete."

# Step 2: Unicharset extraction (loop-safe)
echo "🧩 Extracting unicharset from .box files..."
find "$DATASET_DIR" -name "*.box" -print0 | xargs -0 $ENGINE/unicharset_extractor

# Step 3: Font properties
echo "📝 Creating font_properties..."
echo "arial 0 0 0 0 0" > font_properties

# Step 4: Shape clustering
echo "🔷 Running shape clustering..."
find "$DATASET_DIR" -name "*.tr" -print0 | xargs -0 $ENGINE/shapeclustering -F font_properties -U unicharset

# Step 5: Micro-feature training
echo "⚙️ Running mftraining..."
find "$DATASET_DIR" -name "*.tr" -print0 | xargs -0 $ENGINE/mftraining -F font_properties -U unicharset -O arial.unicharset

# Step 6: Character normalization training
echo "📏 Running cntraining..."
find "$DATASET_DIR" -name "*.tr" -print0 | xargs -0 $ENGINE/cntraining

# Step 7: Combine trained data
echo "📦 Combining traineddata files..."
$ENGINE/combine_tessdata arial.

echo "✅ Training complete! Your model is ready as: arial.traineddata"
