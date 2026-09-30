# src/etl/02_extract_pdf_data.py
import pdfplumber
import camelot
import pandas as pd
from pathlib import Path

class PDFDataExtractor:
    def __init__(self, raw_data_dir='data/raw'):
        self.raw_data_dir = Path(raw_data_dir)
        
    def extract_cag_audit_tables(self, pdf_path):
        """
        Extract tables from CAG Performance Audit Report No. 11 of 2023
        """
        print(f"Extracting tables from {pdf_path}")
        
        try:
            # Method 1: Using camelot (better for structured tables)
            tables = camelot.read_pdf(str(pdf_path), pages='all', flavor='lattice')
            
            all_dataframes = []
            for i, table in enumerate(tables):
                df = table.df
                df.columns = df.iloc[0]  # First row as header
                df = df[1:]  # Remove header row
                
                output_file = self.raw_data_dir / f'cag_table_{i+1}.csv'
                df.to_csv(output_file, index=False)
                print(f"  ✓ Saved table {i+1} to {output_file}")
                all_dataframes.append(df)
            
            return all_dataframes
            
        except Exception as e:
            print(f"  ✗ Camelot extraction failed: {e}")
            
            # Method 2: Using pdfplumber (fallback)
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        tables = page.extract_tables()
                        for j, table in enumerate(tables):
                            df = pd.DataFrame(table[1:], columns=table[0])
                            output_file = self.raw_data_dir / f'cag_page{i+1}_table{j+1}.csv'
                            df.to_csv(output_file, index=False)
                            print(f"  ✓ Saved page {i+1}, table {j+1}")
            except Exception as e2:
                print(f"  ✗ PDFPlumber extraction also failed: {e2}")
    
    def extract_parliamentary_data(self, pdf_path):
        """
        Extract fraud statistics from Parliamentary Q&A PDFs
        """
        print(f"Extracting Parliamentary data from {pdf_path}")
        
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        
        # Save raw text for manual parsing
        output_file = self.raw_data_dir / 'parliamentary_reply_text.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"  ✓ Saved raw text to {output_file}")
        print("  ⚠️  Manual parsing required for structured data extraction")
        
        return text

# Example usage
if __name__ == "__main__":
    extractor = PDFDataExtractor()
    
    # You'll need to manually download these PDFs:
    # 1. CAG Report: https://cag.gov.in (Report No. 11 of 2023)
    # 2. Parliamentary replies from sansad.in
    
    print("⚠️  Place PDF files in data/raw/ directory:")
    print("  - cag_report_2023.pdf")
    print("  - parliament_rajya_sabha_mar2025.pdf")
    print("  - parliament_lok_sabha_dec2025.pdf")