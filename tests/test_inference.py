"""
Comprehensive Test Script for All Inference Modes
Tests single image and multi-view inference pipelines
"""
import os
import sys
import shutil
from PIL import Image
import numpy as np

def create_test_data():
    """Create synthetic test data"""
    print("📦 Creating test data...")
    
    # Create directories
    os.makedirs("test_data/single", exist_ok=True)
    os.makedirs("test_data/multi", exist_ok=True)
    
    # Create synthetic damaged image
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    Image.fromarray(img).save("test_data/single/damaged_artifact.jpg")
    
    # Create multiple images for multi-view
    for i in range(5):
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        Image.fromarray(img).save(f"test_data/multi/view_{i:02d}.jpg")
    
    print("✅ Test data created")
    return True


def test_single_inference():
    """Test single image inference mode"""
    print("\n" + "="*60)
    print("TEST 1: SINGLE IMAGE INFERENCE")
    print("="*60)
    
    cmd = (
        "python main.py "
        "--mode single_inference "
        "--input_image test_data/single/damaged_artifact.jpg "
        "--output_dir test_results/single_test "
        "--no_viz"
    )
    
    print(f"\nRunning: {cmd}\n")
    result = os.system(cmd)
    
    if result == 0:
        # Check outputs
        expected_files = [
            "test_results/single_test/restored.jpg",
            "test_results/single_test/depth_estimated.png",
            "test_results/single_test/mesh_3d.ply"
        ]
        
        all_exist = all(os.path.exists(f) for f in expected_files)
        
        if all_exist:
            print("\n✅ TEST PASSED: All expected files created")
            return True
        else:
            print("\n❌ TEST FAILED: Missing output files")
            for f in expected_files:
                print(f"  {f}: {'✓' if os.path.exists(f) else '✗'}")
            return False
    else:
        print("\n❌ TEST FAILED: Command execution failed")
        return False


def test_multi_inference():
    """Test multi-view inference mode"""
    print("\n" + "="*60)
    print("TEST 2: MULTI-VIEW INFERENCE")
    print("="*60)
    
    cmd = (
        "python main.py "
        "--mode multi_inference "
        "--input_dir test_data/multi "
        "--output_dir test_results/multi_test "
        "--no_viz"
    )
    
    print(f"\nRunning: {cmd}\n")
    result = os.system(cmd)
    
    if result == 0:
        # Check outputs
        expected_dirs = [
            "test_results/multi_test/restored_images",
            "test_results/multi_test/sparse",
            "test_results/multi_test/dense"
        ]
        
        all_exist = all(os.path.isdir(d) for d in expected_dirs)
        
        if all_exist:
            print("\n✅ TEST PASSED: All expected directories created")
            return True
        else:
            print("\n❌ TEST FAILED: Missing output directories")
            for d in expected_dirs:
                print(f"  {d}: {'✓' if os.path.isdir(d) else '✗'}")
            return False
    else:
        print("\n❌ TEST FAILED: Command execution failed")
        return False


def test_help():
    """Test help command"""
    print("\n" + "="*60)
    print("TEST 0: HELP COMMAND")
    print("="*60)
    
    result = os.system("python main.py --help")
    
    if result == 0:
        print("\n✅ TEST PASSED: Help command works")
        return True
    else:
        print("\n❌ TEST FAILED: Help command failed")
        return False


def cleanup():
    """Clean up test data"""
    print("\n🧹 Cleaning up test data...")
    
    if os.path.exists("test_data"):
        shutil.rmtree("test_data")
    if os.path.exists("test_results"):
        shutil.rmtree("test_results")
    
    print("✅ Cleanup complete")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 COMPREHENSIVE INFERENCE TESTING")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists("checkpoints/best_model.pth") and not os.path.exists("checkpoints/last_model.pth"):
        print("\n⚠️ WARNING: No trained model found in checkpoints/")
        print("Please train a model first or these tests will fail.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Setup
    create_test_data()
    
    results = []
    
    # Run tests
    try:
        results.append(("Help Command", test_help()))
        results.append(("Single Inference", test_single_inference()))
        results.append(("Multi Inference", test_multi_inference()))
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test error: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:30s} {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print("="*60)
    print(f"Total: {passed}/{total} tests passed")
    print("="*60)
    
    # Cleanup
    cleanup()
    
    # Exit code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
