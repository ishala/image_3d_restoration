import argparse
from src.visualization.visualize_outputs import visualize_outputs, visualize_comparison, save_screenshot

def main():
    parser = argparse.ArgumentParser(description="Visualize 3D reconstruction results")
    parser.add_argument("--artifact", type=str, required=True,
                        help="Artifact name (e.g., boy_with_thorn)")
    parser.add_argument("--type", type=str, 
                        choices=['original', 'restored', 'compare'],
                        default='restored',
                        help="Which reconstruction to visualize")
    parser.add_argument("--screenshot", type=str,
                        help="Save screenshot to file instead of interactive view")
    args = parser.parse_args()
    
    # Setup paths
    original_dir = f"outputs/{args.artifact}/dense"
    restored_dir = f"outputs/{args.artifact}/reconstruction_from_restored/dense"
    
    # Handle different modes
    if args.type == 'compare':
        visualize_comparison(original_dir, restored_dir, args.artifact)
    else:
        dense_dir = restored_dir if args.type == 'restored' else original_dir
        
        if args.screenshot:
            save_screenshot(dense_dir, args.screenshot)
        else:
            visualize_outputs(dense_dir)

if __name__ == "__main__":
    main()