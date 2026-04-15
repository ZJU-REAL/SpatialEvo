#!/bin/bash
# SpatialEvo quick start script

echo "=========================================="
echo "  SpatialEvo Quick Start"
echo "=========================================="
echo ""

# Check the working directory
if [ ! -f "README.md" ]; then
    echo "❌ Error: run this script from the SpatialEvo project root"
    exit 1
fi

echo "📍 Current directory: $(pwd)"
echo ""

# 1. Check Python
echo "1️⃣  Checking Python..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Error: Python not found"
    exit 1
fi
echo "✅ Python is available"
echo ""

# 2. Check required packages
echo "2️⃣  Checking required packages..."
python3 -c "import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  numpy is missing, installing..."
    pip3 install numpy
fi
echo "✅ Package check complete"
echo ""

# 3. Check data
echo "3️⃣  Checking data..."
METADATA_DIR="/mnt/jfs/lidingm/data/dataset/ScanNet/metadata/scene0000_01"
if [ ! -d "$METADATA_DIR" ]; then
    echo "❌ Error: metadata directory not found: $METADATA_DIR"
    echo "Run the metadata extraction script first:"
    echo "  cd data/scannet_process/data_scripts"
    echo "  python metadata_extractor.py --src_dir ... --dst_dir ..."
    exit 1
fi
echo "✅ Data directory found"
echo ""

# 4. Run the example
echo "4️⃣  Running the example..."
echo ""
python3 examples/vsi_bench_example.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  ✅ Quick start complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Read README.md for the project overview"
    echo "  2. Read VSI_GUIDE.md for detailed usage"
    echo "  3. Read TODO.md for pending work"
    echo ""
    echo "Generate datasets:"
    echo "  cd data/scannet_process/utils"
    echo "  python video_tasks_generator.py \\"
    echo "      --src_dir /mnt/jfs/lidingm/data/dataset/ScanNet/metadata \\"
    echo "      --dst_dir /mnt/jfs/lidingm/data/dataset/ScanNet/tasks"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "  ❌ Example failed"
    echo "=========================================="
    echo ""
    echo "Check the error output and project docs for troubleshooting."
    exit 1
fi
