"""Generate sample product dataset for testing."""
import pandas as pd
import random
from pathlib import Path


# Sample product categories
CATEGORIES = [
    "Running Shoes",
    "Basketball Shoes",
    "Casual Sneakers",
    "Formal Shoes",
    "Hiking Boots",
    "Sandals"
]

# Sample product templates
PRODUCTS = [
    # Running Shoes
    {
        "category": "Running Shoes",
        "templates": [
            "Professional marathon running shoes with advanced cushioning technology",
            "Lightweight trail running shoes designed for long-distance comfort",
            "High-performance running shoes with breathable mesh upper",
            "Carbon fiber plate running shoes for competitive athletes",
            "All-terrain running shoes with superior grip and stability"
        ]
    },
    # Basketball Shoes
    {
        "category": "Basketball Shoes",
        "templates": [
            "High-top basketball shoes with ankle support and responsive cushioning",
            "Signature basketball sneakers with Zoom Air technology",
            "Professional basketball shoes designed for explosive movements",
            "Lightweight basketball shoes with excellent court feel",
            "Retro basketball sneakers with modern performance features"
        ]
    },
    # Casual Sneakers
    {
        "category": "Casual Sneakers",
        "templates": [
            "Classic canvas sneakers perfect for everyday wear",
            "Minimalist white leather sneakers with clean design",
            "Retro-inspired sneakers with vintage colorways",
            "Slip-on sneakers with comfortable memory foam insole",
            "Sustainable sneakers made from recycled materials"
        ]
    },
    # Formal Shoes
    {
        "category": "Formal Shoes",
        "templates": [
            "Premium leather oxford shoes for business professionals",
            "Handcrafted Italian leather dress shoes",
            "Classic black derby shoes with Goodyear welt construction",
            "Monk strap dress shoes with elegant buckle detail",
            "Wingtip brogues with traditional perforation patterns"
        ]
    },
    # Hiking Boots
    {
        "category": "Hiking Boots",
        "templates": [
            "Waterproof hiking boots with Gore-Tex membrane",
            "Lightweight backpacking boots for multi-day treks",
            "Insulated winter hiking boots for cold weather",
            "Mid-cut hiking boots with excellent ankle support",
            "Approach shoes for technical hiking and climbing"
        ]
    },
    # Sandals
    {
        "category": "Sandals",
        "templates": [
            "Comfortable sport sandals with adjustable straps",
            "Leather slide sandals for casual summer wear",
            "Waterproof hiking sandals with rugged outsole",
            "Minimalist flip-flops with arch support",
            "Premium leather sandals with cork footbed"
        ]
    }
]


def generate_sample_data(num_items: int = 100) -> pd.DataFrame:
    """
    Generate sample product data.
    
    Args:
        num_items: Number of items to generate
        
    Returns:
        DataFrame with sample products
    """
    data = []
    
    for i in range(num_items):
        # Select random category
        category_data = random.choice(PRODUCTS)
        category = category_data["category"]
        
        # Select random template
        description = random.choice(category_data["templates"])
        
        # Generate item
        item = {
            "item_id": f"ITEM_{i+1:04d}",
            "title": f"{category} - Model {i+1}",
            "description": description,
            "category": category
        }
        
        data.append(item)
    
    return pd.DataFrame(data)


def main():
    """Generate and save sample data."""
    # Create data directory
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    print("Generating sample product data...")
    df = generate_sample_data(num_items=100)
    
    # Save to CSV
    output_path = data_dir / "sample_products.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} sample products")
    print(f"Saved to: {output_path}")
    print(f"\nCategory distribution:")
    print(df['category'].value_counts())


if __name__ == "__main__":
    main()
