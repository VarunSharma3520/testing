# extract_pdf_to_txt.py
import warnings
from pathlib import Path

# suppress the known harmless pypdf page-label warnings
warnings.filterwarnings("ignore", message="Could not reliably determine page label")

PDF_PATH = Path("./data/F18-ABCD-000.pdf")
OUT_PATH = PDF_PATH.with_suffix(".txt")

# Try the common loaders
loaders = []
# try:
#     from langchain_community.document_loaders import PyPDFLoader
#     loaders.append(("PyPDFLoader", PyPDFLoader))
# except Exception:
#     pass

try:
    from langchain_community.document_loaders import PDFMinerLoader
    loaders.append(("PDFMinerLoader", PDFMinerLoader))
except Exception:
    pass

# optional pdfplumber loader (not part of langchain-community but often available)
try:
    import pdfplumber
    loaders.append(("pdfplumber", "pdfplumber"))
except Exception:
    pass

def load_with_langchain_loader(loader_cls, path):
    loader = loader_cls(str(path))
    return loader.load()

def load_with_pdfplumber(path):
    texts = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            texts.append({"page": i+1, "text": text})
    return texts

def extract_text(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    # Try each loader in order
    last_err = None
    for name, loader in loaders:
        try:
            if name == "pdfplumber":
                raw = load_with_pdfplumber(path)
                # adapt to langchain-like docs: list of objects with page_content and metadata
                docs = []
                for item in raw:
                    docs.append(type("D", (), {"page_content": item["text"], "metadata": {"page": item["page"]}})())
                print(f"[ok] loaded with {name} ({len(docs)} pages)")
                return docs
            else:
                docs = load_with_langchain_loader(loader, path)
                print(f"[ok] loaded with {name} ({len(docs)} pages)")
                return docs
        except Exception as e:
            last_err = e
            print(f"[warn] loader {name} failed: {e}")
            continue

    # If none worked, raise the last error
    raise RuntimeError(f"No loader succeeded. Last error: {last_err}")

def save_text(docs, out_path: Path):
    # Join pages with a clear separator
    parts = []
    for i, d in enumerate(docs):
        page_meta = ""
        meta = getattr(d, "metadata", {}) or {}
        page_meta = f"page:{meta.get('page', meta.get('page_number', i+1))}"
        header = f"\n\n----- {page_meta} -----\n\n"
        content = (d.page_content or "").strip()
        parts.append(header + content)
    full = "\n".join(parts).strip()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(full, encoding="utf-8")
    return out_path, len(parts)

def main():
    docs = extract_text(PDF_PATH)
    out_file, n_pages = save_text(docs, OUT_PATH)
    print(f"Saved text for {n_pages} pages to: {out_file}")

if __name__ == "__main__":
    main()
