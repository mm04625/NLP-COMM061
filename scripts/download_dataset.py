from datasets import load_dataset
import os

def main():
    print("Downloading PLOD-CW-25 dataset...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Load the dataset
    try:
        dataset = load_dataset('surrey-nlp/PLOD-CW-25')
        print("\nDataset successfully loaded!")
        print("\nDataset splits available:", list(dataset.keys()))
        
        # Print basic information about each split
        for split_name, split_data in dataset.items():
            print(f"\n{split_name} split size: {len(split_data)} examples")
            print(f"Features available: {split_data.features}")
            
            # Show first example
            if len(split_data) > 0:
                print(f"\nFirst example from {split_name} split:")
                example = split_data[0]
                for key, value in example.items():
                    print(f"{key}: {value}")
                
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")

if __name__ == "__main__":
    main() 