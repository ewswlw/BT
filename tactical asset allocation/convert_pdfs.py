"""Convert PDF files to text format for analysis."""

import pdfplumber
from pathlib import Path


def convert_pdf_to_text(pdf_path: Path, output_path: Path = None):
    """
    Convert a PDF file to text format.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Optional path for output text file. If None, uses same name with .txt extension
    """
    if output_path is None:
        output_path = pdf_path.with_suffix('.txt')
    
    print(f"Converting {pdf_path.name}...")
    
    all_text = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"  Total pages: {total_pages}")
            
            for i, page in enumerate(pdf.pages, 1):
                if i % 10 == 0:
                    print(f"  Processing page {i}/{total_pages}...")
                
                # Extract text from the page
                text = page.extract_text()
                if text:
                    all_text.append(f"\n{'='*80}\n")
                    all_text.append(f"PAGE {i}\n")
                    all_text.append(f"{'='*80}\n\n")
                    all_text.append(text)
            
            # Write to output file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(''.join(all_text))
            
            print(f"  ✓ Saved to {output_path.name}")
            print(f"  Total characters extracted: {len(''.join(all_text)):,}")
            
    except Exception as e:
        print(f"  ✗ Error converting {pdf_path.name}: {e}")


def main():
    """Convert all PDFs in the research folder."""
    research_dir = Path(__file__).parent / "research"
    
    if not research_dir.exists():
        print(f"Research directory not found: {research_dir}")
        return
    
    # Find all PDF files
    pdf_files = list(research_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in the research directory")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s)\n")
    
    for pdf_file in pdf_files:
        convert_pdf_to_text(pdf_file)
        print()
    
    print("Conversion complete!")


if __name__ == "__main__":
    main()

